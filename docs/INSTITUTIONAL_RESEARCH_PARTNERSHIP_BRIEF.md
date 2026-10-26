# AiMedRes — Institutional Research Partnership Brief

> **Version:** 0.1 (26 October 2026)
> **Audience:** UK university, NHS Trust, research institute and prospective Poland–UK research partners
> **Status:** invitation to shape a research-validation project. This is not a clinical whitepaper, grant application, clinical protocol, or regulatory submission.

## The partnership request

AiMedRes seeks a UK academic-clinical partner to co-design and lead a data-governed, retrospective study of Parkinson's disease progression. The immediate goal is to turn a reproducible **synthetic-fixture research prototype** into independently reviewed evidence on authorised data. The work will not inform diagnosis, treatment, trial enrolment, or patient management.

The proposed first conversation is a 30-minute fit assessment with a clinical PI, a biostatistician or methodologist, and a data-governance representative. It asks whether the institution would be interested in shaping the research question and, if suitable, acting as the eligible lead or co-applicant for a funding proposal.

## Why this is worth evaluating

Parkinson's progression research needs analyses that can be rerun, inspected and challenged before their results are translated. AiMedRes already contains a narrow technical control baseline for that purpose. It is a starting point for an independently governed study, rather than evidence of clinical utility.

| Present in the repository | What it supports | What it does **not** support |
|---|---|---|
| A versioned, non-identifiable synthetic fixture (170 visits, 60 synthetic participants), participant-grouped cross-validation by `PATNO`, fold-contained preprocessing, bootstrap intervals and a provenance manifest. | A locally reproducible software and method-control demonstration. | PPMI, patient, external-cohort, diagnostic, prognostic or clinical-performance claims. |
| A public Evidence Pack with source hashes, command and explicit limitations. | Inspection of one local control run without distributing participant data or model artefacts. | An independent replication, release, signed baseline or controlled-data study. |
| A maintained research API that exposes research-model metadata only; online inference and heuristic-agent output are disabled. | A bounded research surface and a safe synthetic demonstration. | A live prediction service, clinical decision support workflow or neurologist-facing tool. |
| Local security regression controls and source-level response to the October 2026 Render advisory. | A starting point for a partner's technical due diligence. | Penetration-test results, deployed-environment verification, a certified QMS, HIPAA/GDPR certification or NHS deployment assurance. |

The authoritative technical boundaries are the [research-readiness roadmap](roadmap.md), [model card](MODEL_CARD_PPMI_PARKINSONS.md), [Known Limitations](KNOWN_LIMITATIONS.md), and [synthetic Evidence Pack](../evidence/ppmi_synthetic_fixture/2026-10-25/EVIDENCE_PACK.md).

## Draft research question

**Can a preregistered, participant-isolated retrospective analysis of authorised Parkinson's data estimate a clinically agreed, future motor-progression endpoint with acceptable calibration and robustness across temporal or external cohorts?**

The question is deliberately provisional. Before any controlled-data run, the clinical PI, data holder and statistician must agree the endpoint, index time, cohort criteria, missing-data approach, primary metric, subgroup plan and interpretation rule. The current synthetic label is not a clinical endpoint.

## Proposed work packages (24–36 months)

| Work package | Purpose | Key output | Gate before moving on |
|---|---|---|---|
| WP0 — Governance and protocol | Appoint roles; obtain data permissions; define the analysis protocol and PPI involvement. | Preregistered protocol, DUA/data-management plan, roles and approvals appropriate to the study. | Data holder and clinical PI approve the protocol. |
| WP1 — Independent technical replication | Rerun the frozen pipeline in a partner-controlled environment and audit implementation choices. | Independent replication report, including negative or divergent results. | Methodologist signs off that the protocol can be executed as specified. |
| WP2 — Retrospective controlled-data study | Run the prespecified analysis on authorised data without changing claims after inspection. | Cohort manifest, reproducible analysis record, calibration and uncertainty report. | Clinical and statistical review of results and limitations. |
| WP3 — Temporal/external validation | Test transportability on a held-out period or a separate authorised cohort. | External/temporal validation, subgroup and missingness analyses. | Go/no-go decision on whether further usability research is justified. |
| WP4 — Transparent dissemination | Publish the protocol, code where permitted, negative findings and a de-identified evidence summary. | Partner-approved Evidence Pack and dissemination plan. | Funder and data-sharing obligations are met. |

No work package authorises clinical deployment. Any future patient-facing or decision-support research would require a separate intended-use decision, institutional review and evidence plan.

## Partner roles sought

- **UK clinical PI:** clinical relevance, endpoint and cohort definition, governance route and eligible-host role.
- **Biostatistician/methodologist:** prespecified analysis plan, calibration, uncertainty, missing-data and subgroup methodology.
- **Authorised data holder / informatics lead:** DUA, secure environment, data dictionary and reproducible extract procedure.
- **PPI lead:** paid, meaningful involvement of people with lived experience from proposal design through dissemination.
- **AiMedRes technical lead:** reproducible pipeline, provenance controls, synthetic demonstration, documentation and transparent reporting.
- **Optional Poland partner:** only where they have a real research role, eligible costs and compatible IP/data-governance arrangements.

## Funding route and decision logic

| Route | Role in the project | Immediate decision |
|---|---|---|
| **Parkinson's UK Project Grants — primary UK route** | Funds focused Parkinson's projects for up to 36 months. A UK university, research institute, NHS Trust or statutory social-care organisation must host the application; PPI is expected. The 2027 pre-proposal deadline is **7 January 2027, 16:00**. | Proceed once a UK PI, PPI plan, budget owner and protocol outline are agreed. Do not use a synthetic AUC as a clinical-performance claim. |
| **KPFP Fundusz Badań i Wdrożeń 3.0 — conditional regional route** | Potentially funds a tightly scoped Polish research service, external methodological work or IP task. The active Module II/III round closes **12 November 2026**, subject to its rules and possible early closure. | Use only with a genuinely eligible entity and work located in the region; obtain a written eligibility reading before preparing an application. |
| **PARP Ścieżka SMART B+R — conditional Polish SME route** | Potential B+R route for an eligible Polish SME with substantive work in Poland. The announced window is **29 October–29 December 2026**. | Do not pursue through a nominal Polish entity. First confirm SME status, Polish activity, co-finance capacity and a credible innovation/deployment path. |
| **ERA4Health JTC10 / other international calls — watchlist, not a default application** | A possible consortium route if the published topic, partners and national rules match the protocol. The current JTC10 topic is advanced/nano diagnostic and theranostic tools. | Do not force AiMedRes into this call: the current retrospective validation scope is not a demonstrated nanotechnology project. Reassess only after the formal call text and partner fit are reviewed. |
| **NIHR i4i / MJFF — later-stage or call-specific routes** | Potential routes after partner-led proof of principle, a defined clinical need and an active call fit. | Do not submit an unsolicited platform pitch. Check the live call specification and lead institution's eligibility first. |

Official programme pages (checked 26 October 2026): [Parkinson's UK Project Grants](https://www.parkinsons.org.uk/research/funding-grants/project-grants), [KPFP Fundusz Badań i Wdrożeń 3.0](https://kpfp.org.pl/oferta-finansowania/fundusz-badan-i-wdrozen-3-0/), [PARP Ścieżka SMART B+R](https://www.parp.gov.pl/component/grants/grants/sciezka-smart), and [NCBR ERA4Health JTC10](https://www.gov.pl/web/ncbr/nanodiagncd-jtc10-2027---nowy-konkurs-w-ramach-partnerstwa-era4health-start-w-listopadzie-2026-r). The wider UK/Poland opportunity map is maintained in the [funding strategy](FUNDING_STRATEGY_UK_AND_POLAND.md). Funding dates and rules must be checked again immediately before submission.

## A concise pre-proposal narrative (starter text)

**Working title:** *Independent, reproducible validation of retrospective Parkinson's motor-progression modelling across authorised cohorts.*

**Need and opportunity:** Existing AI claims in Parkinson's research are often difficult to reproduce or to assess for participant leakage, calibration and transportability. AiMedRes provides a transparent, research-only pipeline that has been exercised on a non-identifiable synthetic fixture. It has not been validated on PPMI or patient data, and its maintained API intentionally does not provide online inference.

**Aim:** With a UK clinical-academic partner and people affected by Parkinson's, we will preregister and independently replicate a retrospective analysis before applying it to authorised data. We will define a clinician-approved future motor-progression endpoint, isolate participants across folds, fit preprocessing within training folds, report calibration and uncertainty, and test temporal or external transportability. Results, including negative findings, will be reported transparently.

**Benefit:** The project will produce a trustworthy evidence base about whether the approach is suitable for further research. It will not make patient-level clinical recommendations, alter care, or claim regulatory readiness.

**Fields that must be completed by the partner before submission:** named PI and host; endpoint; cohort and DUA; PPI contributors and payment; work-package costs; data-controller roles; IP/publication terms; equality, diversity and inclusion approach; and the funder's current application-specific questions.

## Parkinson's UK pre-proposal readiness sheet

This is a content map for the first pre-proposal discussion. It is **not** a replacement for Parkinson's UK's current online form or applicant guidance.

| Proposal element | AiMedRes draft content | Required partner confirmation |
|---|---|---|
| Project title | *Independent, reproducible validation of retrospective Parkinson's motor-progression modelling across authorised cohorts.* | Confirm scientific wording and that the project does not fall within an excluded call scope. |
| Research gap | Need for transparent, independently reproducible evaluation of Parkinson's progression modelling, including leakage, calibration and transportability controls. | Clinical PI confirms relevance to the current research priority. |
| Objectives | Protocol/governance; independent rerun; authorised retrospective analysis; temporal/external validation; transparent dissemination. | Convert to measurable aims, milestones and success/failure criteria. |
| Design and analysis | Participant-isolated split; fold-contained preprocessing; prespecified primary metric, calibration, uncertainty, missing-data and subgroup analyses. | Biostatistician finalises the SAP before data access/analysis. |
| Data and governance | No controlled data are in the public pack. Controlled-data work starts only after appropriate DUA and institutional governance. | Data holder identifies cohort, controller/processor responsibilities, extraction route and secure environment. |
| PPI and inclusion | PPI is part of WP0 and WP4; it will shape research questions, materials and dissemination. | Name PPI contributors, payment/support plan and EDI approach with the host institution. |
| Outputs and benefit | A transparent evidence base, including negative results, showing whether further research is justified. | Agree dissemination, IP and open-science/data-sharing conditions. |
| Budget and host | Indicative role mix is in the [funding strategy](FUNDING_STRATEGY_UK_AND_POLAND.md). | Host finance team builds a funder-compliant costed application; no budget is inferred from an award ceiling. |
| Clinical boundary | No diagnosis, treatment recommendation, trial enrichment or clinical deployment in this project. | PI confirms the intended use remains research-only throughout the proposal. |

### Submission go/no-go

Do not submit until all of these are true:

1. A UK-eligible host and named clinical PI accept ownership of the application.
2. A statistician approves the draft endpoint and analysis plan.
3. PPI contributors have reviewed the plain-language summary and are resourced appropriately.
4. The relevant data holder has confirmed a viable route to authorised data and governance review.
5. The work packages, costs, IP and publication rights are agreed between partners.
6. The scope remains a focused Parkinson's research question rather than a general AI platform or clinical product launch.
7. The live Parkinson's UK call guidance, eligibility and deadline have been rechecked by the submitting host.

## Ready-to-send first-contact email

**Subject:** Parkinson's research partnership: reproducible retrospective validation proposal

Dear Dr/Prof **[Name]**,

I am developing AiMedRes, a research-only, reproducible analysis prototype for Parkinson's progression research. Its current evidence is deliberately limited: it has been tested on a synthetic fixture only, and its maintained API does not provide clinical or online predictions.

I am looking for a UK clinical-academic partner to co-design a preregistered, retrospective validation study on authorised data. The proposed work would focus on a clinically agreed endpoint, participant-isolated evaluation, calibration, external/temporal validation and meaningful PPI—not a clinical deployment. I have attached a partnership brief and can share the public synthetic Evidence Pack.

Would you be open to a 30-minute conversation with a clinical PI or methodologist to assess scientific fit and eligibility for a Parkinson's UK Project Grant?

Kind regards,
**[Name]**
**[Role / contact details]**

## Before sharing or submitting

1. Send only the brief, roadmap and synthetic Evidence Pack; never share controlled data, secrets, model artefacts or patient-level outputs.
2. Replace all bracketed fields with partner-confirmed information.
3. Have the clinical PI and methodologist review every sentence about endpoint, cohort, performance and intended use.
4. Recheck the funder's call text, eligibility, costs and deadline on the day the proposal is submitted.
5. Treat a partner's interest as exploratory until roles, data governance, IP and funding ownership are agreed in writing.
