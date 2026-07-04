# Family Wiki Implementation Plan

## Overview

Create a MediaWiki-based family wiki with automatically generated pages for people, families, places, and occupations. Each page type will have LLM-generated narratives based on structured data and source text chunks.

## Architecture

### Docker Compose Setup

```yaml
services:
  # Existing services...

  wiki-db:
    image: mariadb:11
    environment:
      MYSQL_ROOT_PASSWORD: ${WIKI_DB_ROOT_PASSWORD}
      MYSQL_DATABASE: wikidb
      MYSQL_USER: wikiuser
      MYSQL_PASSWORD: ${WIKI_DB_PASSWORD}
    volumes:
      - wiki_db_data:/var/lib/mysql
    networks:
      - genealogy-network

  wiki:
    image: mediawiki:1.41
    depends_on:
      - wiki-db
    ports:
      - "8081:80"
    environment:
      MEDIAWIKI_DB_TYPE: mysql
      MEDIAWIKI_DB_SERVER: wiki-db
      MEDIAWIKI_DB_NAME: wikidb
      MEDIAWIKI_DB_USER: wikiuser
      MEDIAWIKI_DB_PASSWORD: ${WIKI_DB_PASSWORD}
      MEDIAWIKI_SITE_NAME: "Family History Wiki"
      MEDIAWIKI_SITE_LANG: en
      MEDIAWIKI_ADMIN_USER: admin
      MEDIAWIKI_ADMIN_PASS: ${WIKI_ADMIN_PASSWORD}
    volumes:
      - wiki_data:/var/www/html/images
      - ./mediawiki/LocalSettings.php:/var/www/html/LocalSettings.php
    networks:
      - genealogy-network

volumes:
  wiki_db_data:
  wiki_data:
```

## MediaWiki Templates

Using templates provides:
- **Consistent formatting** across all pages
- **Easy updates** (change template, all pages update)
- **Semantic structure** for better data extraction
- **Professional appearance** with infoboxes

### Template Strategy

We'll create templates for each page type and common components:

1. **Template:PersonInfobox** - Right-side infobox with key facts
2. **Template:Person** - Full person page structure
3. **Template:FamilyInfobox** - Family group summary box
4. **Template:Timeline** - Reusable timeline component
5. **Template:EventRow** - Row in event tables
6. **Template:SourceRef** - Source document references

## Page Types

### 1. Person Pages

**URL Pattern**: `/Person:{genealogical_id}` (e.g., `/Person:IX.7.b`)

**Using Template**: `{{Person|...}}`

**Template:PersonInfobox** (to create):
```mediawiki
{| class="infobox" style="width: 22em; float: right;"
|-
! colspan="2" style="text-align: center; font-size: larger;" | {{{name}}}
|-
{{#if:{{{image|}}}|
{{!}} colspan="2" style="text-align: center;" {{!}} [[File:{{{image}}}|250px]]
{{!}}-
}}
|-
! Birth
| {{{birth_date}}}{{#if:{{{birth_place|}}}|<br/>{{{birth_place}}}}}
|-
! Death
| {{{death_date}}}{{#if:{{{death_place|}}}|<br/>{{{death_place}}}}}
|-
{{#if:{{{parents|}}}|
{{!}} colspan="2" {{!}} '''Parents'''
{{!}}-
{{!}} colspan="2" {{!}} {{{parents}}}
{{!}}-
}}
|-
! Genealogical ID
| {{{genealogical_id}}}
|-
! Generation
| {{{generation}}}
|}
```

**Content Structure (using templates)**:
```mediawiki
{{PersonInfobox
|name=Eugene Bruce Van Zanten
|birth_date=March 21, 1935
|birth_place=Minneapolis (Hennepin MN)
|death_date=January 1, 2016
|parents=[[Person:VIII.3.d|Pieter Vanzanten]]<br/>[[Person:VIII.3.d.spouse1|Hilda Victoria Fogelberg]]
|genealogical_id=IX.7.b
|generation=IX
}}

== Summary ==
{LLM-generated 2-3 sentence overview}

== Biography ==
{LLM-generated narrative from events and source texts}

== Life Timeline ==
{{Timeline
|events=
{{EventRow|date=1935-03-21|type=Birth|place=Minneapolis (Hennepin MN)}}
{{EventRow|date=1956-12-27|type=Marriage|details=to {spouse}}}
{{EventRow|date=1992|type=Residence|place=Seward (AK)}}
{{EventRow|date=2016-01-01|type=Death}}
}}

== Family ==
=== Parents ===
* [[Person:VIII.3.d|Pieter Vanzanten]] (1908-1973)
* [[Person:VIII.3.d.spouse1|Hilda Victoria Fogelberg]] (1910-1998)

=== Spouses ===
{{#if:{{{spouse1|}}}|
* [[Person:{{{spouse1_id}}}|{{{spouse1_name}}}]] (m. {{{marriage1_date}}})
}}

=== Children ===
{{#if:{{{children|}}}|
{{{children}}}
|''No children recorded''
}}

== Occupations ==
{{#if:{{{occupations|}}}|
{{{occupations}}}
}}

== Residences ==
{{#if:{{{residences|}}}|
{{{residences}}}
}}

== Sources ==
{{SourceRef|document={{{source_doc}}}|page={{{source_page}}}|chunk_id={{{chunk_id}}}}}

[[Category:Person]]
[[Category:Generation {{{generation}}}]]
{{#if:{{{occupation_categories|}}}|
{{#arraymap:{{{occupation_categories}}}|,|x|[[Category:@@@]]|}}
}}
```

**LLM Generation Strategy**:
- **Summary**: Prompt with person name, dates, major life events → 2-3 sentence overview
- **Biography**: Prompt with chronological events + source_texts from chunks → narrative paragraph(s)

### 2. Family Group Pages

**URL Pattern**: `/Family:{generation}.{family_group}` (e.g., `/Family:IX.7`)

**Content Structure**:
```markdown
# Family of {Parent Names} (Generation {roman_numeral})

## Overview
{LLM-generated summary of the family}

## Parents
* [[Person:{parent1_id}|{parent1_name}]] ({birth}-{death})
* [[Person:{parent2_id}|{parent2_name}]] ({birth}-{death})

## Children ({count})
| Name | Born | Died | Occupation | Locations |
|------|------|------|------------|-----------|
| [[Person:IX.7.a|Name]] | {date} | {date} | {occupation} | {places} |
| ... | ... | ... | ... | ... |

## Family Narrative
{LLM-generated narrative about the family dynamics, migrations, occupations}

## Geographic Footprint
* [[Place:{place1}|{place1}]] - {count} family members
* ...

## Occupations Represented
* [[Occupation:Teaching|Teaching]] - {count} family members
* ...

## Timeline
{Combined timeline of all family member events}

[[Category:Family]]
[[Category:Generation_{roman_numeral}]]
```

**LLM Generation Strategy**:
- **Overview**: Prompt with parent names, child count, date range, locations → summary
- **Narrative**: Prompt with all family member events + source texts → family story

### 3. Place Pages

**URL Pattern**: `/Place:{normalized_place_name}` (e.g., `/Place:Minneapolis_MN`)

**Content Structure**:
```markdown
# {Place Name}

## Overview
{LLM-generated summary of family connection to this place}

## Residents Timeline
| Period | Person | Event Type | Details |
|--------|--------|------------|---------|
| 1935-1956 | [[Person:IX.7.b|Eugene Van Zanten]] | Birth, Residence | Born here, lived until marriage |
| ... | ... | ... | ... |

## Family History Narrative
{LLM-generated narrative about the family's relationship to this place}

## Events at This Location
### Births ({count})
* [[Person:{id}|{name}]] ({date})

### Marriages ({count})
* [[Person:{id1}|{name1}]] & [[Person:{id2}|{name2}]] ({date})

### Residences ({count})
* [[Person:{id}|{name}]] ({date_range})

### Deaths ({count})
* [[Person:{id}|{name}]] ({date})

## Geographic Context
{If we have coordinates: embedded map}

[[Category:Place]]
[[Category:Location_{country}]]
```

**LLM Generation Strategy**:
- **Overview**: Prompt with place name, resident count, date range → summary
- **Narrative**: Prompt with all events at location + source texts → place story

### 4. Occupation Pages

**URL Pattern**: `/Occupation:{normalized_occupation}` (e.g., `/Occupation:Building_Trades`)

**Content Structure**:
```markdown
# {Occupation Category}

## Overview
{LLM-generated overview of this occupation in the family}

## People in This Field ({count})
| Name | Period | Specific Role | Location |
|------|--------|---------------|----------|
| [[Person:VI.1.n|Bessel van Zanten]] | 1860s | opzichter der fortificatiën | Naarden |
| ... | ... | ... | ... |

## Historical Context
{LLM-generated narrative about this occupation in Dutch history}

## Family Patterns
{LLM-generated analysis of generational patterns, geographic clusters}

## Related Occupations
* [[Occupation:Carpentry|Carpentry]]
* [[Occupation:Masonry|Masonry]]

## Source Excerpts
{Relevant source text snippets mentioning this occupation}

[[Category:Occupation]]
```

**Occupation Extraction Strategy**:
This is the tricky part. We need to:

1. **Create occupation categories** (manually or LLM-assisted):
   - Building Trades: timmerman, metselaar, opzichter, etc.
   - Military: soldaat, kapitein, ruiter, militair, etc.
   - Musicians: muzikant, organist, zanger, etc.
   - Transportation: koetsier, schipper, bootjesverhuurder, etc.
   - Trades: smid, bakker, slager, etc.
   - Professionals: onderwijzer, arts, advocaat, etc.

2. **Classification approach**:
   ```python
   # Option A: LLM-based classification
   def classify_occupation(occupation_text: str) -> list[str]:
       """
       Use LLM to classify Dutch occupation into categories.

       Prompt: Given this Dutch occupation "{occupation_text}",
               classify it into one or more categories:
               [Building Trades, Military, Musicians, Transportation,
                Trades, Professionals, Agriculture, Maritime, Other]
       """

   # Option B: Keyword matching + LLM fallback
   OCCUPATION_KEYWORDS = {
       'Building Trades': ['timmerman', 'metselaar', 'bouw', 'opzichter', 'aannemer'],
       'Military': ['soldaat', 'kapitein', 'ruiter', 'militair', 'sergeant'],
       'Musicians': ['muzikant', 'organist', 'zanger', 'pianist', 'violist'],
       # ...
   }
   ```

3. **Store classifications**:
   - Add `occupation_categories` JSONField to Event model
   - Run batch classification on all OCCU events
   - Update as new occupations are found

## MediaWiki Template Definitions

We'll need to create these templates in MediaWiki before generating pages:

### Template:PersonInfobox
```mediawiki
<includeonly>{| class="infobox" style="width: 22em; float: right; border: 1px solid #aaa; padding: 5px; background-color: #f9f9f9;"
|-
! colspan="2" style="text-align: center; font-size: larger; background-color: #e0e0e0;" | {{{name}}}
|-
{{#if:{{{image|}}}|
{{!}} colspan="2" style="text-align: center;" {{!}} [[File:{{{image}}}{{!}}250px]]
{{!}}-
}}
|-
! style="text-align: right; padding-right: 10px;" | Birth
| {{{birth_date|}}}{{#if:{{{birth_place|}}}|<br/><small>{{{birth_place}}}</small>}}
|-
! style="text-align: right; padding-right: 10px;" | Death
| {{{death_date|}}}{{#if:{{{death_place|}}}|<br/><small>{{{death_place}}}</small>}}
|-
{{#if:{{{parents|}}}|
! style="text-align: right; padding-right: 10px;" {{!}} Parents
{{!}} {{{parents}}}
{{!}}-
}}
|-
! style="text-align: right; padding-right: 10px;" | Genealogical ID
| {{{genealogical_id}}}
|-
! style="text-align: right; padding-right: 10px;" | Generation
| [[Category:Generation {{{generation}}}]]{{{generation}}}
|}</includeonly><noinclude>
== Usage ==
<pre>
{{PersonInfobox
|name=Eugene Bruce Van Zanten
|birth_date=March 21, 1935
|birth_place=Minneapolis (Hennepin MN)
|death_date=January 1, 2016
|parents=[[Person:VIII.3.d|Pieter Vanzanten]]<br/>[[Person:VIII.3.d.spouse1|Hilda Victoria Fogelberg]]
|genealogical_id=IX.7.b
|generation=IX
}}
</pre>
[[Category:Templates]]
</noinclude>
```

### Template:EventRow
```mediawiki
<includeonly>* '''{{#time:Y-m-d|{{{date}}}}}''': {{{type}}}{{#if:{{{place|}}}| — {{{place}}}}}{{#if:{{{details|}}}| ({{{details}}})}}</includeonly><noinclude>
== Usage ==
<pre>
{{EventRow|date=1935-03-21|type=Birth|place=Minneapolis (Hennepin MN)}}
{{EventRow|date=1956-12-27|type=Marriage|details=to Jane Doe}}
</pre>
[[Category:Templates]]
</noinclude>
```

### Template:SourceRef
```mediawiki
<includeonly><small>Source: [[Document:{{{document}}}|{{{document}}}]], page {{{page}}}{{#if:{{{chunk_id|}}}| (chunk: {{{chunk_id}}})}}</small></includeonly><noinclude>
== Usage ==
<pre>
{{SourceRef|document=Van Zanten Family History|page=63|chunk_id=da296aaa}}
</pre>
[[Category:Templates]]
</noinclude>
```

### Template:FamilyInfobox
```mediawiki
<includeonly>{| class="infobox" style="width: 22em; float: right; border: 1px solid #aaa; padding: 5px; background-color: #f9f9f9;"
|-
! colspan="2" style="text-align: center; font-size: larger; background-color: #e0e0e0;" | {{{family_name}}}
|-
! style="text-align: right; padding-right: 10px;" | Parents
| {{{parents}}}
|-
! style="text-align: right; padding-right: 10px;" | Children
| {{{child_count}}}
|-
! style="text-align: right; padding-right: 10px;" | Generation
| [[Category:Generation {{{generation}}}]]{{{generation}}}
|-
! style="text-align: right; padding-right: 10px;" | Time Period
| {{{time_period}}}
|}</includeonly><noinclude>
== Usage ==
<pre>
{{FamilyInfobox
|family_name=Family of Pieter and Hilda Van Zanten
|parents=[[Person:VIII.3.d|Pieter Vanzanten]] & [[Person:VIII.3.d.spouse1|Hilda Victoria Fogelberg]]
|child_count=7
|generation=IX
|time_period=1935-2016
}}
</pre>
[[Category:Templates]]
</noinclude>
```

## Implementation Phases

### Phase 0: Template Creation (Week 1, Part 1)
- [ ] Create Template:PersonInfobox in MediaWiki
- [ ] Create Template:EventRow in MediaWiki
- [ ] Create Template:SourceRef in MediaWiki
- [ ] Create Template:FamilyInfobox in MediaWiki
- [ ] Create Template:PlaceInfobox in MediaWiki
- [ ] Test templates with sample data
- [ ] Document template parameters

### Phase 1: Infrastructure Setup (Week 1, Part 2)
- [ ] Add MediaWiki + MariaDB to docker-compose.yml
- [ ] Create LocalSettings.php with bot user credentials
- [ ] Set up MediaWiki API client in Django
- [ ] Add template creation method to MediaWikiClient
- [ ] Create Django management command: `generate_wiki`
- [ ] Test basic page creation via API with templates

### Phase 2: Occupation Extraction (Week 2)
- [ ] Define occupation category taxonomy
- [ ] Create occupation classification function (LLM-based)
- [ ] Add `occupation_categories` field to Event model
- [ ] Create migration
- [ ] Create management command: `classify_occupations`
- [ ] Run classification on all existing OCCU events
- [ ] Add tests for occupation classification

### Phase 3: Person Page Generation (Week 3)
- [ ] Create PersonPageGenerator class
  - [ ] Generate timeline from events
  - [ ] LLM: Generate summary (2-3 sentences)
  - [ ] LLM: Generate biography narrative
  - [ ] Collect family links (parents, spouses, children)
  - [ ] Collect occupation links
  - [ ] Collect place links
- [ ] Create MediaWikiClient class
  - [ ] create_page(title, content, summary)
  - [ ] update_page(title, content, summary)
  - [ ] page_exists(title)
- [ ] Generate pages for test family
- [ ] Manual review and iteration

### Phase 4: Family Group Pages (Week 4)
- [ ] Create FamilyPageGenerator class
  - [ ] Parse family groups from genealogical IDs
  - [ ] Generate child table
  - [ ] LLM: Generate family overview
  - [ ] LLM: Generate family narrative
  - [ ] Aggregate geographic footprint
  - [ ] Aggregate occupations
- [ ] Generate pages for test families
- [ ] Manual review and iteration

### Phase 5: Place Pages (Week 5)
- [ ] Create PlacePageGenerator class
  - [ ] Normalize place names (Minneapolis (Hennepin MN) → Minneapolis_MN)
  - [ ] Aggregate all events by place
  - [ ] Create residents timeline
  - [ ] LLM: Generate place overview
  - [ ] LLM: Generate family-place narrative
- [ ] Generate pages for top 20 places
- [ ] Manual review and iteration

### Phase 6: Occupation Pages (Week 6)
- [ ] Create OccupationPageGenerator class
  - [ ] Group people by occupation category
  - [ ] Extract specific roles from OCCU descriptions
  - [ ] LLM: Generate occupation overview
  - [ ] LLM: Generate historical context
  - [ ] LLM: Analyze family patterns
- [ ] Generate pages for all occupation categories
- [ ] Manual review and iteration

### Phase 7: Automation & Updates (Week 7)
- [ ] Create incremental update system
  - [ ] Track last_wiki_update timestamp on models
  - [ ] Only regenerate pages for changed entities
- [ ] Add wiki generation to document processing pipeline
- [ ] Create scheduled task for wiki updates
- [ ] Add wiki links to Django admin

### Phase 8: Polish & Features (Week 8)
- [ ] Add category pages
- [ ] Add navigation templates
- [ ] Add infobox templates for person pages
- [ ] Add search integration
- [ ] Add image upload support (if source images available)
- [ ] Create "Recent Changes" dashboard

## Technical Details

### Django Models

```python
# Add to existing models:

class Event(models.Model):
    # ... existing fields ...
    occupation_categories = models.JSONField(
        default=list,
        blank=True,
        help_text="Occupation categories (e.g., ['Building Trades', 'Military'])"
    )

class Person(models.Model):
    # ... existing fields ...
    wiki_page_title = models.CharField(max_length=255, blank=True)
    wiki_last_updated = models.DateTimeField(null=True, blank=True)

# New model for tracking wiki state:
class WikiPage(models.Model):
    """Track wiki pages generated from database"""

    PAGE_TYPES = [
        ('PERSON', 'Person'),
        ('FAMILY', 'Family'),
        ('PLACE', 'Place'),
        ('OCCUPATION', 'Occupation'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    page_type = models.CharField(max_length=20, choices=PAGE_TYPES)
    page_title = models.CharField(max_length=255, unique=True)

    # Links to source data
    person = models.ForeignKey(Person, null=True, blank=True, on_delete=models.CASCADE)
    # Store family group as JSON: {"generation": 9, "family_group": 7}
    family_group = models.JSONField(null=True, blank=True)
    place_name = models.CharField(max_length=500, blank=True)
    occupation_category = models.CharField(max_length=100, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    needs_regeneration = models.BooleanField(default=False)
```

### MediaWiki API Client

```python
class MediaWikiClient:
    """Client for MediaWiki API operations"""

    def __init__(self, api_url: str, username: str, password: str):
        self.api_url = api_url
        self.session = requests.Session()
        self._login(username, password)

    def create_page(self, title: str, content: str, summary: str = "Auto-generated") -> bool:
        """Create or update a wiki page"""

    def create_template(self, name: str, content: str, summary: str = "Template creation") -> bool:
        """Create a MediaWiki template"""
        return self.create_page(f"Template:{name}", content, summary)

    def page_exists(self, title: str) -> bool:
        """Check if page exists"""

    def get_page_content(self, title: str) -> str:
        """Get current page content"""
```

### Page Generator Example (Using Templates)

```python
class PersonPageGenerator:
    """Generate person pages using MediaWiki templates"""

    def __init__(self, wiki_client: MediaWikiClient, llm_client):
        self.wiki = wiki_client
        self.llm = llm_client

    def generate_page(self, person: Person) -> str:
        """Generate complete person page content using templates"""

        # Get events
        events = person.events.all().order_by('date')
        birth_event = events.filter(event_type='BIRT').first()
        death_event = events.filter(event_type='DEAT').first()

        # Get family
        parents = person.get_parents()
        children = person.get_children()
        spouses = person.get_spouses()

        # Build infobox parameters
        infobox_params = {
            'name': person.full_name,
            'birth_date': birth_event.date.strftime('%B %d, %Y') if birth_event and birth_event.date else 'Unknown',
            'birth_place': birth_event.place if birth_event else '',
            'death_date': death_event.date.strftime('%B %d, %Y') if death_event and death_event.date else '',
            'death_place': death_event.place if death_event else '',
            'genealogical_id': person.genealogical_id,
            'generation': person.generation,
        }

        if parents:
            parent_links = '<br/>'.join([
                f"[[Person:{p.genealogical_id}|{p.full_name}]]"
                for p in parents
            ])
            infobox_params['parents'] = parent_links

        # Generate infobox
        infobox = self._render_template('PersonInfobox', infobox_params)

        # LLM-generate summary
        summary = self.llm.generate_summary(person, events)

        # LLM-generate biography
        biography = self.llm.generate_biography(person, events)

        # Build timeline using EventRow template
        timeline_events = []
        for event in events:
            event_params = {
                'date': event.date.isoformat() if event.date else 'Unknown',
                'type': event.get_event_type_display(),
                'place': event.place or '',
                'details': event.description or '',
            }
            timeline_events.append(self._render_template('EventRow', event_params))

        timeline = '\n'.join(timeline_events)

        # Assemble complete page
        content = f"""{infobox}

== Summary ==
{summary}

== Biography ==
{biography}

== Life Timeline ==
{timeline}

== Family ==
=== Parents ===
{self._format_parent_list(parents)}

=== Spouses ===
{self._format_spouse_list(spouses)}

=== Children ===
{self._format_children_list(children)}

== Sources ==
{self._format_sources(person)}

[[Category:Person]]
[[Category:Generation {person.generation}]]
"""
        return content

    def _render_template(self, template_name: str, params: dict) -> str:
        """Render a MediaWiki template call"""
        param_str = '\n'.join([f"|{k}={v}" for k, v in params.items() if v])
        return f"{{{{{template_name}\n{param_str}\n}}}}"
```

### LLM Prompt Templates

```python
PERSON_SUMMARY_PROMPT = """
Based on this person's life events, write a 2-3 sentence summary:

Name: {full_name}
Born: {birth_date} in {birth_place}
Died: {death_date} in {death_place}
Occupations: {occupations}
Locations: {locations}

Write a concise summary suitable for the opening of a biographical wiki page.
"""

PERSON_BIOGRAPHY_PROMPT = """
Write a biographical narrative for {full_name} based on:

TIMELINE:
{timeline}

SOURCE TEXTS:
{source_texts}

Write 2-4 paragraphs in a narrative style suitable for a wiki biography.
Focus on chronological flow and connecting events into a coherent story.
"""

FAMILY_NARRATIVE_PROMPT = """
Write a family narrative for the {parent_names} family based on:

FAMILY MEMBERS:
{children_summary}

EVENTS:
{family_timeline}

SOURCE TEXTS:
{source_texts}

Write 2-3 paragraphs about this family's story, highlighting patterns in
occupations, migrations, and family dynamics.
"""
```

## Testing Strategy

1. **Unit Tests**:
   - Occupation classification accuracy
   - Place name normalization
   - Timeline generation
   - MediaWiki client methods

2. **Integration Tests**:
   - Generate person page for known individual
   - Generate family page for known group
   - Verify wiki links are correct
   - Test incremental updates

3. **Manual QA**:
   - Generate 10 person pages → review for accuracy
   - Generate 5 family pages → review narratives
   - Generate 5 place pages → verify timelines
   - Generate 3 occupation pages → check categorization

## Open Questions

1. **Occupation taxonomy**: Should we create a hierarchical taxonomy or flat categories?
2. **Place normalization**: How to handle variant spellings (Gameren vs. Gameren (Gelderland))?
3. **Update strategy**: Full regeneration vs. incremental updates?
4. **Images**: Do we have access to any images from source documents we could include?
5. **Version control**: Should we track wiki page history in Django or rely on MediaWiki?

## Future Enhancements

- **Interactive timeline visualization** (using Timeline.js or similar)
- **Family tree visualization** (using D3.js or Graphviz)
- **Map visualization** for places (using Leaflet.js)
- **Source text highlighting** (link to specific chunks)
- **User contributions** (allow manual edits with attribution)
- **Export to GEDCOM** from wiki data
