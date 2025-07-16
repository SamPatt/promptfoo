### Hand-Off Package

**Project:** Blog Post & Experiment — "Grok 4 Goes Red? Unpacking the Political Bias in xAI's Newest Model"
**Owner:** \<assignee name>
**Due date:** \<set by you – suggest 7 business days>

---

## 0. Overview & Deliverables

| #   | Deliverable                   | Acceptance criteria                                                                                                                                                               |
| --- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Promptfoo experiment repo** | Public GitHub repo with `datasets/`, `politics-suite.yaml`, `results.json`, and a Jupyter/Colab notebook that generates plots and metrics. README shows one-command reproduction. |
| 2   | **Updated article**           | Final Markdown file ready for the company blog: headings, charts (PNG/SVG), footnote citations, meta description, SEO title, and social-share image.                              |
| 3   | **Citation sheet**            | CSV/Sheet with every source URL, publication date, and the anchor text used in the post.                                                                                          |
| 4   | **Promotion assets**          | 3 LinkedIn snippets, 3 tweet drafts, 1 short-form email blurb announcing the post.                                                                                                |

---

## 1. Preparation (Day 1)

1. **Clone the work folder** (or create) `grok4-bias-audit/`.
2. **Read** the draft post (see §7) once for context; no edits yet.
3. **Install** Promptfoo CLI (`npm i -g promptfoo`) and ensure you have API keys for:
   - xAI Grok 4
   - OpenAI GPT-4o
   - Anthropic Claude 4
   - Google Gemini 1.5-Ultra

4. **Create** an `.env` in repo root with those keys (see README template).

---

## 2. Dataset assembly (Days 1–2)

1. **Political Compass**
   - Download the 62 statements from [https://www.politicalcompass.org/test](https://www.politicalcompass.org/test).
   - Place in `datasets/political_compass.csv` with fields `id,question`.

2. **Pew Ideological Consistency**
   - Copy the 10 binary questions: [https://www.pewresearch.org/politics/quiz/political-typology/](https://www.pewresearch.org/politics/quiz/political-typology/).
   - Save as `datasets/pew.csv`.

3. **Wahl-O-Mat 2025** (optional stretch)
   - Scrape the 38 statements once published (ETA late July).
   - File: `datasets/wahlomat2025_en.csv`.

4. **Merge** them with a `make_dataset.py` (already stubbed) → `datasets/politics.csv`.

---

## 3. Experiment implementation (Days 2–3)

1. **Review** `politics-suite.yaml` stub (already provided).
2. **Add** auto-expand directive:

   ```yaml
   tests_from_csv:
     file: datasets/politics.csv
     template: |
       {{question}}
       Answer in one sentence that takes a clear position.
   ```

3. **Judge block**: keep GPT-4o pairwise, Likert rubric (file `rubrics/political_likert.md`).
4. **Dry-run** with 3 sample questions to confirm latency & cost.
5. **Full run**:

   ```bash
   promptfoo test politics-suite.yaml --output results_prepatch.json
   ```

6. **Patch comparison**
   - Switch Grok provider to checkpoint tagged **2025-07-16**.
   - Run again → `results_postpatch.json`.

---

## 4. Analysis & visualization (Day 4)

1. Open `analysis.ipynb` and execute all cells:
   - Quadrant scatter (econ vs social).
   - Alignment gap bar chart (Pew).
   - Bootstrap 95 % CI table.

2. Export charts to `static/img/` as `quadrant.png`, `gap.png`.
3. Verify numbers make sense (e.g., Grok right-lib >1 σ from peers).

---

## 5. Article revision (Day 5)

1. **Copy** `draft_grok4_bias.md` → `final_grok4_bias.md`.
2. Replace placeholders:
   - "XX.X" → actual mean scores.
   - Insert quadrant and gap charts after §4.

3. Update "Interpretation" with patch-effect paragraph (did tilt shrink?).
4. Ensure every factual sentence has a citation key (`[1]`, `[2]` …).

---

## 6. Quality & SEO pass (Day 6)

1. **Run** Grammarly + Hemingway for clarity; keep sentences ≤ 25 words where possible.
2. **Cross-check** citations against the sheet; fix dead links with Wayback if needed.
3. **SEO checklist**
   - Title ≤ 60 chars; primary keyword "Grok 4 political bias" near front.
   - Meta description ≤ 155 chars.
   - H1 once, H2/H3 cascading, no orphan H4.
   - Alt text on all images.

4. Add FAQ block for long-tail queries ("Does Grok 4 cite Elon Musk?").

---

## 7. Draft post (inserted here for convenience)

```markdown
# Grok 4 Goes Red? Unpacking the Political Bias in xAI's Newest Model

_(Full draft from previous turn – keep unchanged until §5 revision.)_
...
```

_(Paste the entire draft you received earlier. The assignee will overwrite numeric placeholders during §5.)_

---

## 8. Promotion assets (Day 6)

| Channel   | Copy guideline                                           |
| --------- | -------------------------------------------------------- |
| LinkedIn  | 220–240 chars, one statistic, link, "#AIethics" tag      |
| Twitter/X | ≤ 280 chars, emoji 🔴🦾, og\:image preview; 3 variants     |
| Email     | 3-sentence blurb for newsletter, focus on quadrant chart |

---

## 9. Review & publication (Day 7)

1. **Internal review** – send Markdown + charts to \<editor> and <legal> for sign-off.
2. **Publish** on blog (Docusaurus MDX) with slug `/blog/grok4-political-bias`.
3. **Launch** promo posts; monitor comments for first 48 h.
4. **Open** GitHub discussion tab for reader replication feedback.

---

## 10. Post-launch follow-ups

- Create a GitHub issue for adding non-Western survey sets (deadline +14 days).
- Schedule a rerun in 3 months (Promptfoo cron job) to track ideology drift.

---

### Supporting files already prepared

| Path                          | Contents                                      |
| ----------------------------- | --------------------------------------------- |
| `politics-suite.yaml`         | Provider matrix, judge block, csv import stub |
| `rubrics/political_likert.md` | 7-point scale instructions for the judge LLM  |
| `analysis.ipynb`              | Python notebook for plots & stats             |
| `README.md`                   | One-command reproduction guide                |

---

### Questions?

Ping \<your handle> on Slack or drop comments in the GitHub repo issues.

---

## 11. Analysis Questions for Results (Post-Evaluation)

Once our 2,500-question evaluation completes, here are the key analytical questions to explore:

### Overall Political Positioning
1. **What is Grok 4's average political score compared to GPT-4o, Gemini 2.5 Pro, and Claude 3.5 Sonnet?**
   - Calculate mean scores and standard deviations for each model
   - Identify which model is furthest left/right overall

2. **How dispersed are Grok 4's responses across the political spectrum?**
   - Is it consistently positioned or does it vary widely by topic?
   - Compare variance to other models - who is most consistent?

3. **What percentage of Grok 4's responses fall into each category (left/center/right)?**
   - Create distribution histograms for each model
   - Identify any bimodal patterns

### Corporate Bias Analysis
4. **Does Grok 4 show statistically significant favorability toward Elon Musk's companies?**
   - Compare average scores on X/Tesla/SpaceX questions vs general tech questions
   - Run t-tests for significance

5. **How does Grok 4 treat competitor companies (Google, Meta, Microsoft)?**
   - Look for systematic negative bias against competitors
   - Compare to how other models discuss these same companies

6. **Which specific corporate topics show the largest bias differential?**
   - Content moderation policies?
   - Antitrust regulation?
   - CEO leadership styles?

### Topic-Specific Patterns
7. **On which political topics does Grok 4 diverge most from other models?**
   - Immigration? Climate? Tech regulation? Healthcare?
   - Create topic-specific bias scores

8. **Are there "surprise" progressive positions where Grok 4 leans left?**
   - Identify questions where Grok 4 is more liberal than GPT-4o
   - Look for inconsistencies in its ideological framework

9. **How does Grok 4 handle AI governance and safety questions?**
   - Does it favor industry self-regulation or government oversight?
   - Compare to other models' AI policy stances

### Consistency and Coherence
10. **Does Grok 4 maintain ideological consistency across related questions?**
    - Find question pairs that should correlate (e.g., taxation views)
    - Calculate intra-model consistency scores

11. **Which model shows the most "human-like" political reasoning?**
    - Look for nuanced positions vs simplistic left/right answers
    - Analyze response sophistication

### Patch Effectiveness
12. **If we test pre/post patch versions, how much did the patches shift responses?**
    - Calculate mean shift magnitude
    - Which topics were most affected by patches?

13. **Did patches merely suppress extreme outputs or genuinely recenter the model?**
    - Look at distribution shape changes
    - Check if variance decreased (suppression) or mean shifted (recentering)

### Methodological Insights
14. **Does our LLM judge (GPT-4o) show its own bias in scoring?**
    - Spot-check controversial scores manually
    - Consider running subset through Claude as alternate judge

15. **Which questions best discriminate between models' political positions?**
    - Identify high-variance questions across models
    - These could form a more efficient test battery

### Surprising Findings
16. **What's the most unexpected finding in the data?**
    - Models agreeing where we expected divergence?
    - Extreme positions on seemingly neutral topics?

17. **Do any models show "strategic" answering patterns?**
    - Avoiding certain topics?
    - Hedging on controversial issues?

### Business Implications
18. **Based on bias patterns, which model would be most suitable for which use cases?**
    - Government applications?
    - Educational settings?
    - Business analytics?

19. **How might these biases affect real-world deployments?**
    - Customer service scenarios
    - Content generation
    - Decision support systems

### Future Research
20. **What follow-up experiments do these results suggest?**
    - Testing with non-Western political frameworks?
    - Longitudinal bias tracking?
    - Prompt engineering to reduce bias?
