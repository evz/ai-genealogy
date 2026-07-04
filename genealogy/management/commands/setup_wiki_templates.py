from django.conf import settings
from django.core.management.base import BaseCommand

from genealogy.services.wiki_client import MediaWikiClient


TEMPLATES = {
    "PersonInfobox": """\
<includeonly>{| class="infobox" style="width:22em;float:right;border:1px solid #aaa;padding:5px;background:#faf6f0;margin-left:1em;margin-bottom:1em;"
|-
! colspan="2" style="text-align:center;font-size:1.1em;background:#e8e0d0;padding:6px;" | {{{name}}}
{{#if:{{{birth_date|}}}|
|-
! style="text-align:right;padding-right:8px;white-space:nowrap;" | Born
| {{{birth_date}}}{{#if:{{{birth_place|}}}|<br/><small>{{{birth_place}}}</small>}}
}}
{{#if:{{{bapt_date|}}}|
|-
! style="text-align:right;padding-right:8px;white-space:nowrap;" | Baptised
| {{{bapt_date}}}{{#if:{{{bapt_place|}}}|<br/><small>{{{bapt_place}}}</small>}}
}}
{{#if:{{{death_date|}}}|
|-
! style="text-align:right;padding-right:8px;white-space:nowrap;" | Died
| {{{death_date}}}{{#if:{{{death_place|}}}|<br/><small>{{{death_place}}}</small>}}
}}
{{#if:{{{parents|}}}|
|-
! style="text-align:right;padding-right:8px;white-space:nowrap;vertical-align:top;" | Parents
| {{{parents}}}
}}
{{#if:{{{spouses|}}}|
|-
! style="text-align:right;padding-right:8px;white-space:nowrap;vertical-align:top;" | Spouse(s)
| {{{spouses}}}
}}
|-
! style="text-align:right;padding-right:8px;" | Genealogical ID
| <code>{{{genealogical_id}}}</code>
|-
! style="text-align:right;padding-right:8px;" | Generation
| {{{generation}}}
|}</includeonly><noinclude>
Infobox for person pages.
== Usage ==
<pre>
{{PersonInfobox
|name=Eugene Bruce Van Zanten
|birth_date=21 March 1935
|birth_place=Minneapolis, Minnesota
|death_date=1 January 2016
|parents=[[Person:VIII.3.d|Pieter Vanzanten]]<br/>[[Person:VIII.3.d.spouse1|Hilda Fogelberg]]
|spouses=[[Person:IX.7.b.spouse1|Jane Doe]]
|genealogical_id=IX.7.b
|generation=IX
}}
</pre>
[[Category:Templates]]
</noinclude>""",

    "FamilyInfobox": """\
<includeonly>{| class="infobox" style="width:22em;float:right;border:1px solid #aaa;padding:5px;background:#faf6f0;margin-left:1em;margin-bottom:1em;"
|-
! colspan="2" style="text-align:center;font-size:1.1em;background:#e8e0d0;padding:6px;" | {{{family_name}}}
|-
! style="text-align:right;padding-right:8px;vertical-align:top;" | Parents
| {{{parents}}}
{{#if:{{{marriage_date|}}}|
|-
! style="text-align:right;padding-right:8px;" | Married
| {{{marriage_date}}}{{#if:{{{marriage_place|}}}|<br/><small>{{{marriage_place}}}</small>}}
}}
|-
! style="text-align:right;padding-right:8px;" | Children
| {{{child_count}}}
|-
! style="text-align:right;padding-right:8px;" | Generation
| {{{generation}}}
{{#if:{{{time_period|}}}|
|-
! style="text-align:right;padding-right:8px;" | Period
| {{{time_period}}}
}}
|}</includeonly><noinclude>
Infobox for family group pages.
== Usage ==
<pre>
{{FamilyInfobox
|family_name=Family of Pieter and Hilda Van Zanten
|parents=[[Person:VIII.3.d|Pieter Vanzanten]]<br/>[[Person:VIII.3.d.spouse1|Hilda Fogelberg]]
|marriage_date=1934
|marriage_place=Amsterdam
|child_count=7
|generation=IX
|time_period=1935–2016
}}
</pre>
[[Category:Templates]]
</noinclude>""",

    "PlaceInfobox": """\
<includeonly>{| class="infobox" style="width:22em;float:right;border:1px solid #aaa;padding:5px;background:#faf6f0;margin-left:1em;margin-bottom:1em;"
|-
! colspan="2" style="text-align:center;font-size:1.1em;background:#e8e0d0;padding:6px;" | {{{name}}}
{{#if:{{{region|}}}|
|-
! style="text-align:right;padding-right:8px;" | Region
| {{{region}}}
}}
{{#if:{{{country|}}}|
|-
! style="text-align:right;padding-right:8px;" | Country
| {{{country}}}
}}
|-
! style="text-align:right;padding-right:8px;" | Family members
| {{{person_count}}}
{{#if:{{{date_range|}}}|
|-
! style="text-align:right;padding-right:8px;" | Active period
| {{{date_range}}}
}}
|}</includeonly><noinclude>
Infobox for place pages.
== Usage ==
<pre>
{{PlaceInfobox
|name=Amsterdam
|region=North Holland
|country=Netherlands
|person_count=12
|date_range=1820–1950
}}
</pre>
[[Category:Templates]]
</noinclude>""",

    "OccupationInfobox": """\
<includeonly>{| class="infobox" style="width:22em;float:right;border:1px solid #aaa;padding:5px;background:#faf6f0;margin-left:1em;margin-bottom:1em;"
|-
! colspan="2" style="text-align:center;font-size:1.1em;background:#e8e0d0;padding:6px;" | {{{category}}}
|-
! style="text-align:right;padding-right:8px;" | People
| {{{person_count}}}
{{#if:{{{date_range|}}}|
|-
! style="text-align:right;padding-right:8px;" | Period
| {{{date_range}}}
}}
{{#if:{{{example_roles|}}}|
|-
! style="text-align:right;padding-right:8px;vertical-align:top;" | Example roles
| {{{example_roles}}}
}}
|}</includeonly><noinclude>
Infobox for occupation category pages.
== Usage ==
<pre>
{{OccupationInfobox
|category=Building Trades
|person_count=8
|date_range=1750–1900
|example_roles=timmerman, metselaar, aannemer
}}
</pre>
[[Category:Templates]]
</noinclude>""",

    "EventRow": """\
<includeonly>* '''{{{date|}}}'''{{#if:{{{type|}}}| — {{{type}}}}}{{#if:{{{place|}}}| in {{{place}}}}}{{#if:{{{details|}}}|: {{{details}}}}}</includeonly><noinclude>
One line in a life events list.
== Usage ==
<pre>
{{EventRow|date=21 March 1935|type=Birth|place=Minneapolis, Minnesota}}
{{EventRow|date=27 December 1956|type=Marriage|details=to Jane Doe}}
{{EventRow|date=1992|type=Residence|place=Seward, Alaska}}
</pre>
[[Category:Templates]]
</noinclude>""",

    "SourceQuote": """\
<includeonly><div class="source-quote" style="border-left:3px solid #c8b89a;padding:0.4em 0.8em;margin:0.6em 0 0.6em 1em;background:#fdf8f0;">
<div style="font-style:italic;">{{{text}}}</div>
{{#if:{{{translation|}}}|
<div style="margin-top:0.4em;color:#444;"><small>'''Translation:''' {{{translation}}}</small></div>
}}
<div style="text-align:right;margin-top:0.3em;"><small>&#8212; [[Document:{{{document}}}|{{{document}}}]], p.&#160;{{{page}}}</small></div>
</div></includeonly><noinclude>
Displays a quoted source passage with optional English translation and document citation.
Intended for embedding source evidence directly in biographical pages.
== Usage ==
<pre>
{{SourceQuote
|text=Den 21 Maart 1935 is geboren Eugene Bruce, zoon van Pieter Vanzanten...
|translation=On 21 March 1935 was born Eugene Bruce, son of Pieter Vanzanten...
|document=Van Zanten Family History
|page=63
}}
</pre>
[[Category:Templates]]
</noinclude>""",
}

CATEGORIES = {
    "Person": "All people in the Van Zanten family history.\n\n[[Category:Index]]",
    "Family": "Family group pages covering parents and their children.\n\n[[Category:Index]]",
    "Place": "Geographic locations connected to the Van Zanten family.\n\n[[Category:Index]]",
    "Occupation": "Occupation categories found across the family history.\n\n[[Category:Index]]",
    "Document": "Source documents used as evidence in this wiki.\n\n[[Category:Index]]",
    "Templates": "MediaWiki templates used for consistent page formatting.",
    "Index": "Top-level index categories.",
}


class Command(BaseCommand):
    help = "Create or update MediaWiki templates and base categories."

    def add_arguments(self, parser):
        parser.add_argument(
            "--wiki-url",
            default=getattr(settings, "WIKI_API_URL", "http://localhost:8081/api.php"),
            help="MediaWiki API endpoint",
        )
        parser.add_argument(
            "--wiki-user",
            default=getattr(settings, "WIKI_ADMIN_USER", "admin"),
        )
        parser.add_argument("--wiki-pass", default=getattr(settings, "WIKI_ADMIN_PASS", ""))

    def handle(self, *args, **options):
        client = MediaWikiClient(
            api_url=options["wiki_url"],
            username=options["wiki_user"],
            password=options["wiki_pass"],
        )

        self.stdout.write("Creating templates...")
        for name, content in TEMPLATES.items():
            result = client.create_template(name, content)
            if "error" in result:
                self.stderr.write(self.style.ERROR(f"  Template:{name} — {result['error']}"))
            else:
                self.stdout.write(self.style.SUCCESS(f"  Template:{name} — ok"))

        self.stdout.write("Creating categories...")
        for name, description in CATEGORIES.items():
            result = client.create_category_page(name, description)
            if "error" in result:
                self.stderr.write(self.style.ERROR(f"  Category:{name} — {result['error']}"))
            else:
                self.stdout.write(self.style.SUCCESS(f"  Category:{name} — ok"))

        self.stdout.write(self.style.SUCCESS("Done."))
