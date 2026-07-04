import re

from django.conf import settings
from django.core.management.base import BaseCommand

from genealogy.models import Person, TextChunk
from genealogy.ollama_utils import OllamaClient
from genealogy.services.wiki_client import MediaWikiClient

SYSTEM_PROMPT = """You are a genealogical historian specializing in Dutch and Dutch-American family history.
You write clear, engaging biographical summaries in English for a family history wiki.
When source texts are in Dutch, translate and incorporate them naturally.
Use a warm but factual tone. Do not use markdown headers or XML tags in your response."""

BIOGRAPHY_PROMPT = """Write a wiki biography for {full_name} ({genealogical_id}).

KNOWN FACTS:
{event_summary}

SOURCE TEXTS (may be in Dutch or English):
{source_texts}

Respond using exactly this format — no deviations:

SUMMARY:
[2-3 sentence overview of this person's life suitable for a wiki introduction]

BIOGRAPHY:
[2-4 paragraphs of narrative biography, integrating the source information]

TRANSLATION:
[A concise English translation of the primary Dutch source text, preserving the genealogical detail. If the source is already in English, write "Source is in English."]"""


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_llm_response(text: str) -> dict:
    """Extract SUMMARY, BIOGRAPHY, TRANSLATION sections from LLM output."""
    text = strip_thinking(text)
    result = {"summary": "", "biography": "", "translation": ""}

    for key in ("summary", "biography", "translation"):
        pattern = rf"{key.upper()}:\s*(.*?)(?=(?:SUMMARY|BIOGRAPHY|TRANSLATION):|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    return result


def sanitize_wiki_title(name: str) -> str:
    # MediaWiki forbids: # < > [ ] | { }
    # [ ] appear as nickname markers in source data — convert to parens
    name = name.replace("[", "(").replace("]", ")")
    # Strip any remaining forbidden chars
    for ch in "#<>|{}":
        name = name.replace(ch, "")
    return name.strip()


def person_wiki_title(person: Person) -> str:
    name = sanitize_wiki_title(person.full_name)
    return f"Person:{name} ({person.genealogical_id})"


def get_family_group_prefix(genealogical_id: str) -> str:
    """Extract the family group prefix from a genealogical ID.

    'VI.1.n' -> 'VI.1.'   (matches 'VI.1. Kinderen van...' in family_groups)
    'VIII.3.d.spouse1' -> 'VIII.3.'
    """
    parts = genealogical_id.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2]) + "."
    return genealogical_id


# Cache citation chunks so we don't re-query 390 times
_CITATION_CHUNKS_CACHE: list[TextChunk] | None = None


def get_citation_chunks_for_person(person: Person) -> list[TextChunk]:
    """Find source_citation chunks whose family_groups match this person's family group."""
    global _CITATION_CHUNKS_CACHE
    if _CITATION_CHUNKS_CACHE is None:
        _CITATION_CHUNKS_CACHE = list(
            TextChunk.objects.filter(chunk_type="source_citation").select_related("document")
        )

    prefix = get_family_group_prefix(person.genealogical_id)
    return [
        chunk for chunk in _CITATION_CHUNKS_CACHE
        if any(fg.startswith(prefix) for fg in chunk.family_groups)
    ]


def get_unique_chunks(person: Person) -> list[TextChunk]:
    chunk_ids = set()
    chunks = []
    for event in person.events.select_related("source_chunk__document").all():
        if event.source_chunk_id and event.source_chunk_id not in chunk_ids:
            chunk_ids.add(event.source_chunk_id)
            chunks.append(event.source_chunk)
    return chunks


def build_event_summary(person: Person) -> str:
    lines = []
    for e in person.events.order_by("date", "date_original"):
        parts = [f"[{e.event_type}]"]
        if e.date_original:
            parts.append(e.date_original)
        if e.place:
            parts.append(f"— {e.place}")
        if e.description:
            parts.append(f"({e.description})")
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "No events recorded."


def build_event_timeline(person: Person) -> str:
    lines = []
    for e in person.events.order_by("date", "date_original"):
        date_str = e.date.strftime("%-d %B %Y") if e.date else (e.date_original or "unknown date")
        event_type = e.get_event_type_display()
        place = e.place or ""
        desc = e.description or ""
        details = desc if desc and desc.lower() not in (event_type.lower(), place.lower()) else ""
        lines.append(f"{{{{EventRow|date={date_str}|type={event_type}|place={place}|details={details}}}}}")
    return "\n".join(lines)


ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n: int) -> str:
    result = ""
    for value, numeral in ROMAN:
        while n >= value:
            result += numeral
            n -= value
    return result


def format_person_link(p: Person) -> str:
    return f"[[{person_wiki_title(p)}|{p.full_name}]]"


def build_infobox(person, birth, death, bapt, parents, spouses) -> str:
    """Generate a raw wiki table infobox — avoids {{#if:}} pipe-escaping issues."""

    def fmt(e):
        if not e:
            return ""
        return e.date.strftime("%-d %B %Y") if e.date else (e.date_original or "")

    def row(label, value):
        return f'|-\n! style="text-align:right;padding-right:8px;white-space:nowrap;vertical-align:top;" | {label}\n| {value}'

    generation = to_roman(person.generation) if person.generation else "?"
    lines = [
        '{| class="infobox" style="width:22em;float:right;border:1px solid #aaa;padding:5px;background:#faf6f0;margin-left:1em;margin-bottom:1em;"',
        f'|-\n! colspan="2" style="text-align:center;font-size:1.1em;background:#e8e0d0;padding:6px;" | {person.full_name}',
    ]
    if birth:
        cell = fmt(birth)
        if birth.place:
            cell += f"<br/><small>{birth.place}</small>"
        lines.append(row("Born", cell))
    if bapt:
        cell = fmt(bapt)
        if bapt.place:
            cell += f"<br/><small>{bapt.place}</small>"
        lines.append(row("Baptised", cell))
    if death:
        cell = fmt(death)
        if death.place:
            cell += f"<br/><small>{death.place}</small>"
        lines.append(row("Died", cell))
    if parents:
        lines.append(row("Parents", "<br/>".join(format_person_link(p) for p in parents)))
    if spouses:
        lines.append(row("Spouse(s)", "<br/>".join(format_person_link(s) for s in spouses)))
    lines.append(row("Genealogical ID", f"<code>{person.genealogical_id}</code>"))
    lines.append(row("Generation", generation))
    lines.append("|}")
    return "\n".join(lines)


def build_person_page(person: Person, biography: str, summary: str, chunks: list[TextChunk], translation: str, citation_chunks: list[TextChunk] | None = None) -> str:
    events = list(person.events.order_by("date", "date_original"))
    birth = next((e for e in events if e.event_type == "BIRT"), None)
    death = next((e for e in events if e.event_type == "DEAT"), None)
    bapt = next((e for e in events if e.event_type == "BAPT"), None)

    parents = [r.parent for r in person.parent_relationships.select_related("parent").all()]
    spouses_p1 = [pt.partner2 for pt in person.partnerships_as_partner1.select_related("partner2").all()]
    spouses_p2 = [pt.partner1 for pt in person.partnerships_as_partner2.select_related("partner1").all()]
    spouses = spouses_p1 + spouses_p2
    children = [r.child for r in person.children_relationships.select_related("child").all()]

    infobox = build_infobox(person, birth, death, bapt, parents, spouses)

    timeline = build_event_timeline(person)

    family_sections = []
    if parents:
        family_sections.append("=== Parents ===\n" + "\n".join(f"* {format_person_link(p)}" for p in parents))
    if spouses:
        family_sections.append("=== Spouse(s) ===\n" + "\n".join(f"* {format_person_link(s)}" for s in spouses))
    if children:
        family_sections.append("=== Children ===\n" + "\n".join(f"* {format_person_link(c)}" for c in children))
    family_block = "\n\n".join(family_sections)

    # Build source quotes
    source_quotes = []
    for i, chunk in enumerate(chunks):
        doc_title = chunk.document.title
        page_num = chunk.start_page
        orig_text = chunk.text_content.replace("|", "{{!}}")  # escape wiki pipe char
        trans = translation if i == 0 else ""
        quote = (
            "{{SourceQuote\n"
            f"|text={orig_text}\n"
            + (f"|translation={trans}\n" if trans and trans != "Source is in English." else "")
            + f"|document={doc_title}\n"
            f"|page={page_num}\n"
            "}}"
        )
        source_quotes.append(quote)
    sources_block = "\n\n".join(source_quotes)

    # Append family group citations (archive references for this family group)
    if citation_chunks:
        citation_quotes = []
        for chunk in citation_chunks:
            orig = chunk.text_content.replace("|", "{{!}}")
            citation_quotes.append(
                "{{SourceQuote\n"
                f"|text={orig}\n"
                f"|document={chunk.document.title}\n"
                f"|page={chunk.start_page}\n"
                "}}"
            )
        if citation_quotes:
            sources_block += "\n\n=== Family Group Citations ===\n" + "\n\n".join(citation_quotes)

    generation = to_roman(person.generation) if person.generation else "?"
    categories = f"[[Category:Person]]\n[[Category:Generation {generation}]]"

    page = f"""{infobox}

== Summary ==
{summary or "''No summary available.''"}

== Biography ==
{biography or "''No biography available.''"}

== Life Timeline ==
{timeline}

== Family ==
{family_block}

== Sources ==
{sources_block}

{categories}"""

    return page


class Command(BaseCommand):
    help = "Seed MediaWiki with person pages generated from the genealogy database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--wiki-url",
            default=getattr(settings, "WIKI_API_URL", "http://host.docker.internal:8081/api.php"),
        )
        parser.add_argument("--wiki-user", default=getattr(settings, "WIKI_ADMIN_USER", "admin"))
        parser.add_argument("--wiki-pass", default=getattr(settings, "WIKI_ADMIN_PASS", ""))
        parser.add_argument("--model", default="deepseek-r1:32b")
        parser.add_argument("--person-id", help="Only generate page for this genealogical_id")
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            default=True,
            help="Skip pages that already exist in the wiki (default: on)",
        )
        parser.add_argument(
            "--no-skip-existing",
            dest="skip_existing",
            action="store_false",
            help="Regenerate pages even if they already exist",
        )

    def handle(self, *args, **options):
        wiki = MediaWikiClient(
            api_url=options["wiki_url"],
            username=options["wiki_user"],
            password=options["wiki_pass"],
        )
        ollama = OllamaClient()

        if not ollama.is_available():
            self.stderr.write(self.style.ERROR("Ollama is not reachable. Check OLLAMA_HOST/OLLAMA_PORT."))
            return

        model = options["model"]
        self.stdout.write(f"Using model: {model}")

        if options["person_id"]:
            people = Person.objects.filter(genealogical_id=options["person_id"])
        else:
            people = Person.objects.all().order_by("genealogical_id")

        total = people.count()
        self.stdout.write(f"Generating pages for {total} people...")

        for i, person in enumerate(people, 1):
            title = person_wiki_title(person)

            if options["skip_existing"] and wiki.page_exists(title):
                self.stdout.write(f"  [{i}/{total}] {title} — skipped (exists)")
                continue

            self.stdout.write(f"  [{i}/{total}] {title} — generating...", ending="")
            self.stdout.flush()

            chunks = get_unique_chunks(person)

            citation_chunks = get_citation_chunks_for_person(person)

            if not chunks and not citation_chunks:
                self.stdout.write(self.style.WARNING(" no sources, using stub."))
                parsed = {"summary": "", "biography": "", "translation": ""}
            else:
                all_source_texts = [c.text_content for c in chunks]
                if citation_chunks:
                    all_source_texts += [c.text_content for c in citation_chunks]
                source_texts = "\n\n---\n\n".join(all_source_texts)
                event_summary = build_event_summary(person)

                prompt = BIOGRAPHY_PROMPT.format(
                    full_name=person.full_name,
                    genealogical_id=person.genealogical_id,
                    event_summary=event_summary,
                    source_texts=source_texts,
                )

                raw = ollama.generate(
                    model=model,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    options={"temperature": 0.3, "num_ctx": 8192},
                )

                if not raw:
                    self.stdout.write(self.style.WARNING(" LLM returned nothing, using stub."))
                    parsed = {"summary": "", "biography": "", "translation": ""}
                else:
                    parsed = parse_llm_response(raw)

            content = build_person_page(
                person=person,
                biography=parsed["biography"],
                summary=parsed["summary"],
                chunks=chunks,
                translation=parsed["translation"],
                citation_chunks=citation_chunks,
            )

            result = wiki.create_or_update_page(title, content, summary=f"Auto-generated biography")

            if "error" in result:
                self.stdout.write(self.style.ERROR(f" error: {result['error']}"))
            else:
                self.stdout.write(self.style.SUCCESS(" done"))

        self.stdout.write(self.style.SUCCESS(f"\nFinished. {total} people processed."))
