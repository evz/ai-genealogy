from django.conf import settings
from django.core.management.base import BaseCommand

from genealogy.models import Person, Archive
from genealogy.management.commands.seed_wiki import person_wiki_title, to_roman, get_unique_chunks
from genealogy.services.wiki_client import MediaWikiClient


FEATURED_ID = "VI.1.n"  # Bessel van Zanten — rich story, good Dutch/English mix


def build_main_page(people_count: int, generation_range: tuple, featured: Person, archive_list: list) -> str:
    gen_min, gen_max = generation_range
    gen_min_r = to_roman(gen_min)
    gen_max_r = to_roman(gen_max)

    featured_title = person_wiki_title(featured)
    birth = featured.events.filter(event_type="BIRT").first()
    death = featured.events.filter(event_type="DEAT").first()
    birth_str = birth.date.strftime("%-d %B %Y") if birth and birth.date else (birth.date_original if birth else "")
    death_str = death.date.strftime("%-d %B %Y") if death and death.date else (death.date_original if death else "")
    lifespan = f"{birth_str} – {death_str}" if birth_str or death_str else ""

    # Generation index: one entry per generation
    from django.db.models import Count
    gen_rows = []
    for g in range(gen_min, gen_max + 1):
        count = Person.objects.filter(generation=g).count()
        if count:
            gen_rows.append(
                f"| [[Category:Generation {to_roman(g)}|Generation {to_roman(g)}]] "
                f"|| {count} people"
            )
    generation_table = (
        '{| class="wikitable"\n'
        "|-\n! Generation !! People\n|-\n"
        + "\n|-\n".join(gen_rows)
        + "\n|}"
    )

    # Archive table
    if archive_list:
        archive_rows = []
        for a in archive_list:
            website_cell = f"[{a.website} {a.abbreviation}]" if a.website else a.abbreviation
            archive_rows.append(f"| {website_cell} || {a.name} || {a.city}")
        archive_table = (
            '{| class="wikitable"\n'
            "|-\n! Code !! Archive !! City\n|-\n"
            + "\n|-\n".join(archive_rows)
            + "\n|}"
        )
    else:
        archive_table = "''Archive list not yet available.''"

    page = (
        "__NOTOC__\n"
        '{| style="width:100%; border:none; padding:0;"\n'
        "|-\n"
        '| style="width:65%; padding-right:2em; vertical-align:top;" |\n\n'
        "== Welcome ==\n"
        f"This wiki documents the genealogy of the Van Zanten family, tracing {people_count} individuals "
        f"across {gen_max - gen_min + 1} generations (Generation {gen_min_r} through {gen_max_r}). "
        "It is based on original research from primary Dutch archives, civil records, church registers, and family sources.\n\n"
        "Pages are generated from structured genealogical data and sourced from primary documents. "
        "Family members are encouraged to add context, correct errors, and upload photographs.\n\n"
        "; Browse by generation\n"
        f"{generation_table}\n\n"
        "; [[Category:Place|Browse by place]] &nbsp;·&nbsp; [[Category:Occupation|Browse by occupation]] "
        "&nbsp;·&nbsp; [[Special:Random|Random family member]]\n\n"
        '| style="width:35%; vertical-align:top; border-left:1px solid #ddd; padding-left:1.5em;" |\n\n'
        f"== Featured: [[{featured_title}|{featured.full_name}]] ==\n"
        f"{lifespan}\n\n"
        f"{featured.full_name} served in the Dutch military as an overseer of fortifications before becoming "
        "a railway crossing keeper and later an elementary schoolteacher — a life that spanned three careers "
        "and reflected the social upheavals of 19th-century Netherlands.\n\n"
        f"[[{featured_title}|Read more...]]\n\n"
        "----\n\n"
        "== Source Archives ==\n"
        "The source documents for this wiki cite the following Dutch archives. Full contact details will be added over time.\n\n"
        f"{archive_table}\n\n"
        "|}\n\n"
        "[[Category:Index]]"
    )
    return page


class Command(BaseCommand):
    help = "Create or update the MediaWiki main page."

    def add_arguments(self, parser):
        parser.add_argument(
            "--wiki-url",
            default=getattr(settings, "WIKI_API_URL", "http://host.docker.internal:8081/api.php"),
        )
        parser.add_argument("--wiki-user", default=getattr(settings, "WIKI_ADMIN_USER", "admin"))
        parser.add_argument("--wiki-pass", default=getattr(settings, "WIKI_ADMIN_PASS", ""))
        parser.add_argument("--featured-id", default=FEATURED_ID)

    def handle(self, *args, **options):
        wiki = MediaWikiClient(
            api_url=options["wiki_url"],
            username=options["wiki_user"],
            password=options["wiki_pass"],
        )

        people = Person.objects.all()
        people_count = people.count()
        generations = people.exclude(generation=None).values_list("generation", flat=True)
        gen_min = min(generations)
        gen_max = max(generations)

        try:
            featured = Person.objects.get(genealogical_id=options["featured_id"])
        except Person.DoesNotExist:
            self.stderr.write(f"Featured person {options['featured_id']} not found, using first person.")
            featured = Person.objects.exclude(generation=None).order_by("genealogical_id").first()

        archives = list(Archive.objects.all().order_by("abbreviation"))

        content = build_main_page(
            people_count=people_count,
            generation_range=(gen_min, gen_max),
            featured=featured,
            archive_list=archives,
        )

        result = wiki.create_or_update_page("Main_Page", content, summary="Auto-generated main page")
        if "error" in result:
            self.stderr.write(self.style.ERROR(f"Failed: {result['error']}"))
        else:
            self.stdout.write(self.style.SUCCESS(f"Main page updated ({people_count} people, generations {to_roman(gen_min)}–{to_roman(gen_max)})."))
