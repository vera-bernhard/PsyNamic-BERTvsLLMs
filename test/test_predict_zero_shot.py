import unittest
from zero_shot.predict_zero_shot import parse_class_prediction, get_label2int, parse_ner_prediction


class TestPredictZeroShot(unittest.TestCase):

    # add a data for the entire class
    def setUp(self):
        self.task = 'Study Type'
        self.label2int = get_label2int(self.task)

    def test_parse_class_llama2(self):
        model = "Llama-2-13-chat-hf,"
        input_data = '''
        [INST] <<SYS>>
You are a helpful medical expert who is helping to classify medical abstracts.
<</SYS>>
***TASK***

The task is to classify the study type based on the title and abstract of a research paper.

***INPUT***

The input is the title and abstract text.

***DOCUMENTATION***

There are 9 Study Type options. The followings are the options and their definitions:

Randomized-controlled trial (RCT): A study where individuals are randomly assigned to one of two or more groups, with one receiving a psychedelic substance and the others receiving a placebo or standard treatment, to compare outcomes directly. I.e., the study needs to check to attributes: 1) being randomized and 2) being controlled (e.g., with a placebo or an active control).

Cohort study: An observational study where a defined group of people (the cohort) is followed over time to study outcomes related to an exposure to psychedelic substance(s), comparing results within the group or with a different group not exposed to the substance. In most cases, it is explicitly mentioned that this is a cohort study.

Real-world study: Examines the relationship between variables (e.g., the use of a psychedelic substance and mental health outcomes) at a single point across a real-world population and setting. This includes real-world studies on recreational drug use, survey-based studies e.g., on MDMA use among healthy adults or studies assessing emergency department visits of people ingesting MDMA. Studies recruiting recreational drug users for additional interventions or experimental procedures such as taking blood or conducting EEG do NOT belong to this category.

Study protocol: A protocol of a clinical trial outlining methodology of a study still to be conducted. Annotate as if the study has been conducted.

Systematic review/meta-analysis: A comprehensive summary of all the available primary research on a specific psychedelic-related research question. Should mention the term systematic review and/or meta-analysis in the title/abstract. Meta-analyses only refer to published studies, i.e., pooling of different RCTs would be “Other”.

Qualitative Study: Explores subjective experiences, perceptions, and the impact of psychedelic substances through interviews, focus groups, or content analysis. Also includes studies focusing on religion/spirituality related to psychedelic substances.

Case report: A detailed report of the symptoms, signs, diagnosis, treatment, and follow-up of one individual patient/healthy participant.

Case series: A collection of case reports involving patients who were given similar interventions.

Other: A study not belonging to one of the above categories or not classifiable based on the provided information. A study pooling different RCTs together belongs to this category.

***OUTPUT***

The output should be in a json format, with relevant value for each option: Randomized-controlled trial (RCT), Cohort study, Real-world study, Study protocol, Systematic review/meta-analysis, Qualitative Study, Case report, Case series, Other

Put value 1 if the option applies to the research paper, 0 if it does not apply.

Please note again that a single option can be selected for each research paper.

Example output format:
{
    ""Randomized-controlled trial (RCT)"": """",
    ""Cohort study"": """",
    ""Real-world study"": """",
    ""Study protocol"": """",
    ""Systematic review/meta-analysis"": """",
    ""Qualitative Study"": """",
    ""Case report"": """",
    ""Case series"": """",
    ""Other"": """"
}


INPUT: A Multi-Site Study of MDMA-Assisted Psychotherapy for PTSD.^\nPTSD is a serious debilitating disorder that negatively impacts a person's daily life. PTSD is a stress‐related psychiatric condition that may occur following a traumatic event such as war, disaster, sexual abuse, violence, terrorism, and accidents. PTSD negatively impacts a person's daily life, resulting in relationship difficulties, difficulty in finding and maintaining a job, reduced cognitive and psychosocial functioning, substance abuse, high‐cost healthcare use, and increased depression and suicide risk. Available PTSD treatments, including medications and therapy, effectively treat only a fraction of people who try them for adequate dose and duration. People with PTSD can be treated with psychotherapies and pharmacotherapies. In the past decade, there has been a growing amount of research into medications and other methods that may augment the effectiveness of psychotherapy for PTSD 3,4‐methylenedioxymethamphetamine (MDMA) is a drug that releases serotonin, norepinephrine and dopamine in the brain and indirectly increases levels of the neurohormones oxytocin, arginine vasopressin and cortisol. The combined neurobiological effects of MDMA increase compassion, reduce defenses and fear of emotional injury, and enhance communication and introspection. MDMA produces anxiolytic and prosocial effects, which counteract avoidance and hyperarousal in the context of therapy. A combined treatment of MDMA and psychotherapy may be especially useful for treating PTSD. This multi‐site, double‐blind, randomized Phase 3 study assesses the efficacy and safety of MDMA‐assisted psychotherapy versus psychotherapy with placebo control in participants diagnosed with at least severe posttraumatic stress disorder (PTSD). The study will be conducted in N ≈ 100 participants. Participants will be enrolled in one of two groups at a 1:1 ratio. A flexible dose of MDMA or placebo, followed by a supplemental half‐dose unless contraindicated, is administered during the Treatment Period with manualized psychotherapy in three monthly Experimental Sessions. This 12‐week Treatment Period is preceded by three Preparatory Sessions. During the Treatment Period, each Experimental Session is followed by three Integrative Sessions of non‐drug psychotherapy. The Primary Outcome measure is change in Clinician Administered PTSD Scale for DSM 5 (CAPS‐5) from Baseline. Exploratory measures will address specific symptoms, or behavior that is sometimes related to PTSD. Drug safety will be assessed by measuring blood pressure, heart rate and body temperature during experimental sessions, collecting adverse events and measuring suicidal thoughts or behaviors with the Columbia Suicide Severity Rating Scale (CSSRS). This study will compare the effects of three manualized Experimental Sessions of psychotherapy supported by assisted by flexible doses of MDMA versus placebo. Initial doses per Experimental Session include 80 mg or 120 mg of MDMA compounded with mannitol and magnesium stearate or placebo alone (mannitol and magnesium stearate), followed 1.5 to 2 hours later by a supplemental half‐dose (40 or 60 mg). Total amounts of MDMA to be administered per Experimental Session range from 80 mg to 180


OUTPUT: [/INST]  Based on the input provided, the output would be:

{
""Randomized-controlled trial (RCT)"": 1,
""Cohort study"": 0,
""Real-world study"": 0,
""Study protocol"": 0,
""Systematic review/meta-analysis"": 0,
""Qualitative Study"": 0,
""Case report"": 0,
""Case series"": 0,
""Other"": 0
}

Explanation:

* Randomized-controlled trial (RCT): The study is a multi-site, double-blind, randomized Phase 3 study assessing the efficacy and safety of MDMA-assisted psychotherapy versus placebo control in participants diagnosed with at least severe posttraumatic stress disorder (PTSD). Therefore, the value for this option is 1.
* Cohort study: The study does not involve a cohort of individuals being followed over time, so the value for this option is 0.
* Real-world study: The study is not set in a real-world setting, as it is a controlled clinical trial, so the value for this option is 0.
* Study protocol: The study is not a protocol for a future study, so the value for this option is 0.
* Systematic review/meta-analysis: The study is not a systematic review or meta-analysis, so the value for this option is 0.
* Qualitative Study: The study does not explore subjective experiences, perceptions, or the impact of psychedelic substances through interviews, focus groups, or content analysis, so the value for this option is 0.
* Case report: The study is not a detailed report of the symptoms, signs, diagnosis, treatment, and follow-up of one individual patient/healthy participant, so the value for this option is 0.
* Case series: The study is not a collection of case reports involving patients who were given similar interventions, so the value for this option is 0.
* Other: The study does not belong to any of the other categories, so the value for this option is 0.'''
        expected = "[1, 0, 0, 0, 0, 0, 0, 0, 0]"

        result = parse_class_prediction(input_data, self.label2int, model)
        self.assertEqual(result, expected)

    def test_parse_class_mellama2(self):
        model = "MeLLaMA-13B-chat,"
        input_data = '''***TASK***

The task is to classify the study type based on the title and abstract of a research paper.

***INPUT***

The input is the title and abstract text.

***DOCUMENTATION***

There are 9 Study Type options. The followings are the options and their definitions:

Randomized-controlled trial (RCT): A study where individuals are randomly assigned to one of two or more groups, with one receiving a psychedelic substance and the others receiving a placebo or standard treatment, to compare outcomes directly. I.e., the study needs to check to attributes: 1) being randomized and 2) being controlled (e.g., with a placebo or an active control).

Cohort study: An observational study where a defined group of people (the cohort) is followed over time to study outcomes related to an exposure to psychedelic substance(s), comparing results within the group or with a different group not exposed to the substance. In most cases, it is explicitly mentioned that this is a cohort study.

Real-world study: Examines the relationship between variables (e.g., the use of a psychedelic substance and mental health outcomes) at a single point across a real-world population and setting. This includes real-world studies on recreational drug use, survey-based studies e.g., on MDMA use among healthy adults or studies assessing emergency department visits of people ingesting MDMA. Studies recruiting recreational drug users for additional interventions or experimental procedures such as taking blood or conducting EEG do NOT belong to this category.

Study protocol: A protocol of a clinical trial outlining methodology of a study still to be conducted. Annotate as if the study has been conducted.

Systematic review/meta-analysis: A comprehensive summary of all the available primary research on a specific psychedelic-related research question. Should mention the term systematic review and/or meta-analysis in the title/abstract. Meta-analyses only refer to published studies, i.e., pooling of different RCTs would be “Other”.

Qualitative Study: Explores subjective experiences, perceptions, and the impact of psychedelic substances through interviews, focus groups, or content analysis. Also includes studies focusing on religion/spirituality related to psychedelic substances.

Case report: A detailed report of the symptoms, signs, diagnosis, treatment, and follow-up of one individual patient/healthy participant.

Case series: A collection of case reports involving patients who were given similar interventions.

Other: A study not belonging to one of the above categories or not classifiable based on the provided information. A study pooling different RCTs together belongs to this category.

***OUTPUT***

The output should be in a json format, with relevant value for each option: Randomized-controlled trial (RCT), Cohort study, Real-world study, Study protocol, Systematic review/meta-analysis, Qualitative Study, Case report, Case series, Other

Put value 1 if the option applies to the research paper, 0 if it does not apply.

Please note again that a single option can be selected for each research paper.

Example output format:
{
    ""Randomized-controlled trial (RCT)"": """",
    ""Cohort study"": """",
    ""Real-world study"": """",
    ""Study protocol"": """",
    ""Systematic review/meta-analysis"": """",
    ""Qualitative Study"": """",
    ""Case report"": """",
    ""Case series"": """",
    ""Other"": """"
}


INPUT: Anxiolytic effects of acute and maintenance ketamine, as assessed by the Fear Questionnaire subscales and the Spielberger State Anxiety Rating Scale.^\nBACKGROUND: Ketamine has rapid anxiolytic effects in treatment-resistant obsessive compulsive, post-traumatic stress, generalised anxiety and social anxiety disorders. OBJECTIVES: This study aimed to assess changes following acute and maintenance ketamine therapy on the Fear Questionnaire (FQ) subscales and the Spielberger State Anxiety Inventory (SSAI). METHODS: This secondary analysis used data from a mixed open-label and double-blinded placebo-controlled study. A total of 24 patients received short-term ascending subcutaneous doses of ketamine and were then eligible to enter a 3-month maintenance phase of 1 mg/kg ketamine dosed once or twice weekly. FQ and SSAI data were analysed using mixed models to identify between-dose differences and to describe trends during maintenance. RESULTS: Acute ketamine dosing showed a rapid dose-related reduction in all three FQ subscales (agoraphobia, social phobia and blood-injury phobia) and in the SSAI. A progressive decrease in pre-dose rating-scale scores was evident during the 3 months of maintenance therapy. CONCLUSIONS: Ketamine demonstrated dose-related improvements in all FQ subscales and in the SSAI. Both scales appear to be suitable tools to assess the anxiolytic effects of ketamine in patients with treatment-resistant anxiety. Furthermore, ketamine appears to have broad, dose-related anti-phobic effects. These findings raise the possibility that ketamine may have therapeutic potential in the treatment of other phobic states, such as specific phobia.


OUTPUT:
 {
    ""Randomized-controlled trial (RCT)"": 0,
    ""Cohort study"": 0,
    ""Real-world study"": 0,
    ""Study protocol"": 0,
    ""Systematic review/meta-analysis"": 0,
    ""Qualitative Study"": 0,
    ""Case report"": 0,
    ""Case series"": 0,
    ""Other"": 1
}
'''
        expected = "[0, 0, 0, 0, 0, 0, 0, 0, 1]"
        results = parse_class_prediction(input_data, self.label2int, model)

    def test_parse_class_mellama2_faulty(self):
        model = "MeLLaMA-13B-chat,"
        input_data = '''***TASK***

The task is to classify the study type based on the title and abstract of a research paper.

***INPUT***

The input is the title and abstract text.

***DOCUMENTATION***

There are 9 Study Type options. The followings are the options and their definitions:

Randomized-controlled trial (RCT): A study where individuals are randomly assigned to one of two or more groups, with one receiving a psychedelic substance and the others receiving a placebo or standard treatment, to compare outcomes directly. I.e., the study needs to check to attributes: 1) being randomized and 2) being controlled (e.g., with a placebo or an active control).

Cohort study: An observational study where a defined group of people (the cohort) is followed over time to study outcomes related to an exposure to psychedelic substance(s), comparing results within the group or with a different group not exposed to the substance. In most cases, it is explicitly mentioned that this is a cohort study.

Real-world study: Examines the relationship between variables (e.g., the use of a psychedelic substance and mental health outcomes) at a single point across a real-world population and setting. This includes real-world studies on recreational drug use, survey-based studies e.g., on MDMA use among healthy adults or studies assessing emergency department visits of people ingesting MDMA. Studies recruiting recreational drug users for additional interventions or experimental procedures such as taking blood or conducting EEG do NOT belong to this category.

Study protocol: A protocol of a clinical trial outlining methodology of a study still to be conducted. Annotate as if the study has been conducted.

Systematic review/meta-analysis: A comprehensive summary of all the available primary research on a specific psychedelic-related research question. Should mention the term systematic review and/or meta-analysis in the title/abstract. Meta-analyses only refer to published studies, i.e., pooling of different RCTs would be “Other”.

Qualitative Study: Explores subjective experiences, perceptions, and the impact of psychedelic substances through interviews, focus groups, or content analysis. Also includes studies focusing on religion/spirituality related to psychedelic substances.

Case report: A detailed report of the symptoms, signs, diagnosis, treatment, and follow-up of one individual patient/healthy participant.

Case series: A collection of case reports involving patients who were given similar interventions.

Other: A study not belonging to one of the above categories or not classifiable based on the provided information. A study pooling different RCTs together belongs to this category.

***OUTPUT***

The output should be in a json format, with relevant value for each option: Randomized-controlled trial (RCT), Cohort study, Real-world study, Study protocol, Systematic review/meta-analysis, Qualitative Study, Case report, Case series, Other

Put value 1 if the option applies to the research paper, 0 if it does not apply.

Please note again that a single option can be selected for each research paper.

Example output format:
{
    ""Randomized-controlled trial (RCT)"": """",
    ""Cohort study"": """",
    ""Real-world study"": """",
    ""Study protocol"": """",
    ""Systematic review/meta-analysis"": """",
    ""Qualitative Study"": """",
    ""Case report"": """",
    ""Case series"": """",
    ""Other"": """"
}

    
INPUT: A Multi-Site Study of MDMA-Assisted Psychotherapy for PTSD.^\nPTSD is a serious debilitating disorder that negatively impacts a person's daily life. PTSD is a stress‐related psychiatric condition that may occur following a traumatic event such as war, disaster, sexual abuse, violence, terrorism, and accidents. PTSD negatively impacts a person's daily life, resulting in relationship difficulties, difficulty in finding and maintaining a job, reduced cognitive and psychosocial functioning, substance abuse, high‐cost healthcare use, and increased depression and suicide risk. Available PTSD treatments, including medications and therapy, effectively treat only a fraction of people who try them for adequate dose and duration. People with PTSD can be treated with psychotherapies and pharmacotherapies. In the past decade, there has been a growing amount of research into medications and other methods that may augment the effectiveness of psychotherapy for PTSD 3,4‐methylenedioxymethamphetamine (MDMA) is a drug that releases serotonin, norepinephrine and dopamine in the brain and indirectly increases levels of the neurohormones oxytocin, arginine vasopressin and cortisol. The combined neurobiological effects of MDMA increase compassion, reduce defenses and fear of emotional injury, and enhance communication and introspection. MDMA produces anxiolytic and prosocial effects, which counteract avoidance and hyperarousal in the context of therapy. A combined treatment of MDMA and psychotherapy may be especially useful for treating PTSD. This multi‐site, double‐blind, randomized Phase 3 study assesses the efficacy and safety of MDMA‐assisted psychotherapy versus psychotherapy with placebo control in participants diagnosed with at least severe posttraumatic stress disorder (PTSD). The study will be conducted in N ≈ 100 participants. Participants will be enrolled in one of two groups at a 1:1 ratio. A flexible dose of MDMA or placebo, followed by a supplemental half‐dose unless contraindicated, is administered during the Treatment Period with manualized psychotherapy in three monthly Experimental Sessions. This 12‐week Treatment Period is preceded by three Preparatory Sessions. During the Treatment Period, each Experimental Session is followed by three Integrative Sessions of non‐drug psychotherapy. The Primary Outcome measure is change in Clinician Administered PTSD Scale for DSM 5 (CAPS‐5) from Baseline. Exploratory measures will address specific symptoms, or behavior that is sometimes related to PTSD. Drug safety will be assessed by measuring blood pressure, heart rate and body temperature during experimental sessions, collecting adverse events and measuring suicidal thoughts or behaviors with the Columbia Suicide Severity Rating Scale (CSSRS). This study will compare the effects of three manualized Experimental Sessions of psychotherapy supported by assisted by flexible doses of MDMA versus placebo. Initial doses per Experimental Session include 80 mg or 120 mg of MDMA compounded with mannitol and magnesium stearate or placebo alone (mannitol and magnesium stearate), followed 1.5 to 2 hours later by a supplemental half‐dose (40 or 60 mg). Total amounts of MDMA to be administered per Experimental Session range from 80 mg to 180


OUTPUT: 1'''

        expected = None
        results = parse_class_prediction(input_data, self.label2int, model)
        self.assertEqual(expected, None)

    def test_parse_class_gpt(self):
        model = 'gpt-4o-mini-2024-07-18'
        input_data = '"{\n    ""Randomized-controlled trial (RCT)"": ""1"",\n    ""Cohort study"": ""0"",\n    ""Real-world study"": ""0"",\n    ""Study protocol"": ""0"",\n    ""Systematic review/meta-analysis"": ""0"",\n    ""Qualitative Study"": ""0"",\n    ""Case report"": ""0"",\n    ""Case series"": ""0"",\n    ""Other"": ""0""\n}'
        excepted = "[1, 0, 0, 0, 0, 0, 0, 0, 0]"
        results = parse_class_prediction(input_data, self.label2int, model)

    def test_parse_class_gpt_faulty(self):
        model = 'gpt-4o-mini-2024-07-18'
        input_data = '''"{
    ""Randomized-controlled trial (RCT)"": """",
    ""Cohort study"": """",
    ""Real-world study"": """",
    ""Study protocol"": """",
    ""Systematic review/meta-analysis"": ""1"",
    ""Qualitative Study"": """",
    ""Case report"": """",
    ""Case series"": """",
    ""Other"": """"
}"'''
        excepted = "[0, 0, 0, 0, 1, 0, 0, 0, 0]"
        results = parse_class_prediction(input_data, self.label2int, model)
        self.assertEqual(excepted, results)

    def test_parse_class_gpt_faulty2(self):
        model = 'gpt-4o-mini-2024-07-18'
        input_data = '{\\n    "Randomized-controlled trial (RCT)": ",\\n    "Cohort study": ",\\n    "Real-world study": ",\\n    "Study protocol": ",\\n    "Systematic review/meta-analysis": "1",\\n    "Qualitative Study": ",\\n    "Case report": ",\\n    "Case series": ",\\n    "Other": "\\n}'
        excepted = "[0, 0, 0, 0, 1, 0, 0, 0, 0]"
        results = parse_class_prediction(input_data, self.label2int, model)
        self.assertEqual(excepted, results)

    def test_ner_parse_gpt(self):
        model = 'gpt-40-2024-08-06'
        pred_data = '''"```html
            Default Mode Connectivity in <span class=""application-area"">Major Depressive Disorder</span> Measured Up to 10 Days After Ketamine Administration.^
            BACKGROUND: The symptoms of <span class=""application-area"">major depressive disorder (MDD)</span> are rapidly alleviated by administration of a single dose of the glutamatergic modulator ketamine. However, few studies have investigated the potential sustained neural effects of this agent beyond immediate infusion. This study used functional magnetic resonance imaging to examine the effect of a single ketamine infusion on the resting state default mode network (DMN) at 2 and 10 days after a single ketamine infusion in unmedicated subjects with <span class=""application-area"">MDD</span> as well as healthy control subjects (HCs). METHODS: Data were drawn from a double-blind, placebo-controlled crossover study of 58 participants (33 with <span class=""application-area"">MDD</span> and 25 HCs) who received an intravenous infusion of either ketamine hydrochloride (<span class=""dosage"">0.5 mg/kg</span>) or placebo on 2 separate test days spaced 2 weeks apart. Eight minutes of functional magnetic resonance imaging resting state data was acquired at baseline and at about 2 and 10 days after both infusions. The DMN was defined using seed-based correlation and was compared across groups and scans. RESULTS: In subjects with <span class=""application-area"">MDD</span>, connectivity between the insula and the DMN was normalized compared with HCs 2 days postketamine infusion. This change was reversed after 10 days and did not appear in either of the placebo scans. Group-specific connectivity differences in drug response were observed, most notably in the insula in subjects with <span class=""application-area"">MDD</span> and in the thalamus in HCs. CONCLUSIONS: Connectivity changes in the insula in subjects with <span class=""application-area"">MDD</span> suggest that ketamine may normalize the interaction between the DMN and salience networks, supporting the triple network dysfunction model of <span class=""application-area"">MDD</span>.
            ```"'''
        tokens = ['Default', 'Mode', 'Connectivity', 'in', 'Major', 'Depressive', 'Disorder', 'Measured', 'Up', 'to', '10', 'Days', 'After', 'Ketamine', 'Administration.^', '\n', 'BACKGROUND', ':', 'The', 'symptoms', 'of', 'major', 'depressive', 'disorder', '(', 'MDD', ')', 'are', 'rapidly', 'alleviated', 'by', 'administration', 'of', 'a', 'single', 'dose', 'of', 'the', 'glutamatergic', 'modulator', 'ketamine', '.', 'However', ',', 'few', 'studies', 'have', 'investigated', 'the', 'potential', 'sustained', 'neural', 'effects', 'of', 'this', 'agent', 'beyond', 'immediate', 'infusion', '.', 'This', 'study', 'used', 'functional', 'magnetic', 'resonance', 'imaging', 'to', 'examine', 'the', 'effect', 'of', 'a', 'single', 'ketamine', 'infusion', 'on', 'the', 'resting', 'state', 'default', 'mode', 'network', '(', 'DMN', ')', 'at', '2', 'and', '10', 'days', 'after', 'a', 'single', 'ketamine', 'infusion', 'in', 'unmedicated', 'subjects', 'with', 'MDD', 'as', 'well', 'as', 'healthy', 'control', 'subjects', '(', 'HCs', ')', '.', 'METHODS', ':', 'Data', 'were', 'drawn', 'from', 'a', 'double', '-', 'blind', ',', 'placebo', '-', 'controlled', 'crossover', 'study', 'of', '58', 'participants', '(', '33', 'with', 'MDD', 'and', '25', 'HCs', ')', 'who', 'received', 'an', 'intravenous', 'infusion', 'of', 'either', 'ketamine', 'hydrochloride', '(', '0.5', 'mg', '/', 'kg', ')', 'or',
                  'placebo', 'on', '2', 'separate', 'test', 'days', 'spaced', '2', 'weeks', 'apart', '.', 'Eight', 'minutes', 'of', 'functional', 'magnetic', 'resonance', 'imaging', 'resting', 'state', 'data', 'was', 'acquired', 'at', 'baseline', 'and', 'at', 'about', '2', 'and', '10', 'days', 'after', 'both', 'infusions', '.', 'The', 'DMN', 'was', 'defined', 'using', 'seed', '-', 'based', 'correlation', 'and', 'was', 'compared', 'across', 'groups', 'and', 'scans', '.', 'RESULTS', ':', 'In', 'subjects', 'with', 'MDD', ',', 'connectivity', 'between', 'the', 'insula', 'and', 'the', 'DMN', 'was', 'normalized', 'compared', 'with', 'HCs', '2', 'days', 'postketamine', 'infusion', '.', 'This', 'change', 'was', 'reversed', 'after', '10', 'days', 'and', 'did', 'not', 'appear', 'in', 'either', 'of', 'the', 'placebo', 'scans', '.', 'Group', '-', 'specific', 'connectivity', 'differences', 'in', 'drug', 'response', 'were', 'observed', ',', 'most', 'notably', 'in', 'the', 'insula', 'in', 'subjects', 'with', 'MDD', 'and', 'in', 'the', 'thalamus', 'in', 'HCs', '.', 'CONCLUSIONS', ':', 'Connectivity', 'changes', 'in', 'the', 'insula', 'in', 'subjects', 'with', 'MDD', 'suggest', 'that', 'ketamine', 'may', 'normalize', 'the', 'interaction', 'between', 'the', 'DMN', 'and', 'salience', 'networks', ',', 'supporting', 'the', 'triple', 'network', 'dysfunction', 'model', 'of', 'MDD', '.']

        expected = ['O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                    'O', 'O', 'O', 'B-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O']

        result = parse_ner_prediction(pred_data, tokens, model)
        self.assertEqual(result, expected)

    def test_ner_parse_gpt_mini(self):
        model = 'gpt-4o-mini-06-06-06'
        pred = '''```html
<h1>Default Mode Connectivity in <span class=""application-area"">Major Depressive Disorder</span> Measured Up to <span class=""dosage"">10 Days</span> After <span class=""dosage"">Ketamine Administration</span>.</h1>
<p>BACKGROUND: The symptoms of <span class=""application-area"">major depressive disorder</span> (MDD) are rapidly alleviated by administration of a single dose of the glutamatergic modulator ketamine. However, few studies have investigated the potential sustained neural effects of this agent beyond immediate infusion. This study used functional magnetic resonance imaging to examine the effect of a single ketamine infusion on the resting state default mode network (DMN) at <span class=""dosage"">2</span> and <span class=""dosage"">10</span> days after a single ketamine infusion in unmedicated subjects with <span class=""application-area"">MDD</span> as well as healthy control subjects (HCs). METHODS: Data were drawn from a double-blind, placebo-controlled crossover study of <span class=""dosage"">58</span> participants (<span class=""dosage"">33</span> with <span class=""application-area"">MDD</span> and <span class=""dosage"">25</span> HCs) who received an intravenous infusion of either ketamine hydrochloride (<span class=""dosage"">0.5 mg/kg</span>) or placebo on <span class=""dosage"">2</span> separate test days spaced <span class=""dosage"">2</span> weeks apart. Eight minutes of functional magnetic resonance imaging resting state data was acquired at baseline and at about <span class=""dosage"">2</span> and <span class=""dosage"">10</span> days after both infusions. The DMN was defined using seed-based correlation and was compared across groups and scans. RESULTS: In subjects with <span class=""application-area"">MDD</span>, connectivity between the insula and the DMN was normalized compared with HCs <span class=""dosage"">2</span> days postketamine infusion. This change was reversed after <span class=""dosage"">10</span> days and did not appear in either of the placebo scans. Group-specific connectivity differences in drug response were observed, most notably in the insula in subjects with <span class=""application-area"">MDD</span> and in the thalamus in HCs. CONCLUSIONS: Connectivity changes in the insula in subjects with <span class=""application-area"">MDD</span> suggest that ketamine may normalize the interaction between the DMN and salience networks, supporting the triple network dysfunction model of <span class=""application-area"">MDD</span>.</p>
```'''
        tokens = ['Default', 'Mode', 'Connectivity', 'in', 'Major', 'Depressive', 'Disorder', 'Measured', 'Up', 'to', '10', 'Days', 'After', 'Ketamine', 'Administration.^', '\n', 'BACKGROUND', ':', 'The', 'symptoms', 'of', 'major', 'depressive', 'disorder', '(', 'MDD', ')', 'are', 'rapidly', 'alleviated', 'by', 'administration', 'of', 'a', 'single', 'dose', 'of', 'the', 'glutamatergic', 'modulator', 'ketamine', '.', 'However', ',', 'few', 'studies', 'have', 'investigated', 'the', 'potential', 'sustained', 'neural', 'effects', 'of', 'this', 'agent', 'beyond', 'immediate', 'infusion', '.', 'This', 'study', 'used', 'functional', 'magnetic', 'resonance', 'imaging', 'to', 'examine', 'the', 'effect', 'of', 'a', 'single', 'ketamine', 'infusion', 'on', 'the', 'resting', 'state', 'default', 'mode', 'network', '(', 'DMN', ')', 'at', '2', 'and', '10', 'days', 'after', 'a', 'single', 'ketamine', 'infusion', 'in', 'unmedicated', 'subjects', 'with', 'MDD', 'as', 'well', 'as', 'healthy', 'control', 'subjects', '(', 'HCs', ')', '.', 'METHODS', ':', 'Data', 'were', 'drawn', 'from', 'a', 'double', '-', 'blind', ',', 'placebo', '-', 'controlled', 'crossover', 'study', 'of', '58', 'participants', '(', '33', 'with', 'MDD', 'and', '25', 'HCs', ')', 'who', 'received', 'an', 'intravenous', 'infusion', 'of', 'either', 'ketamine', 'hydrochloride', '(', '0.5', 'mg', '/', 'kg', ')', 'or',
                  'placebo', 'on', '2', 'separate', 'test', 'days', 'spaced', '2', 'weeks', 'apart', '.', 'Eight', 'minutes', 'of', 'functional', 'magnetic', 'resonance', 'imaging', 'resting', 'state', 'data', 'was', 'acquired', 'at', 'baseline', 'and', 'at', 'about', '2', 'and', '10', 'days', 'after', 'both', 'infusions', '.', 'The', 'DMN', 'was', 'defined', 'using', 'seed', '-', 'based', 'correlation', 'and', 'was', 'compared', 'across', 'groups', 'and', 'scans', '.', 'RESULTS', ':', 'In', 'subjects', 'with', 'MDD', ',', 'connectivity', 'between', 'the', 'insula', 'and', 'the', 'DMN', 'was', 'normalized', 'compared', 'with', 'HCs', '2', 'days', 'postketamine', 'infusion', '.', 'This', 'change', 'was', 'reversed', 'after', '10', 'days', 'and', 'did', 'not', 'appear', 'in', 'either', 'of', 'the', 'placebo', 'scans', '.', 'Group', '-', 'specific', 'connectivity', 'differences', 'in', 'drug', 'response', 'were', 'observed', ',', 'most', 'notably', 'in', 'the', 'insula', 'in', 'subjects', 'with', 'MDD', 'and', 'in', 'the', 'thalamus', 'in', 'HCs', '.', 'CONCLUSIONS', ':', 'Connectivity', 'changes', 'in', 'the', 'insula', 'in', 'subjects', 'with', 'MDD', 'suggest', 'that', 'ketamine', 'may', 'normalize', 'the', 'interaction', 'between', 'the', 'DMN', 'and', 'salience', 'networks', ',', 'supporting', 'the', 'triple', 'network', 'dysfunction', 'model', 'of', 'MDD', '.']

        expected = ['O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'B-Dosage', 'I-Dosage', 'O', 'B-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'O', 'B-Dosage', 'O', 'B-Application area', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                    'B-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O']
        result = parse_ner_prediction(pred, tokens, model)
        # for r, p, t in zip(result, expected, tokens):
        #     print(r, p, t)
        self.assertEqual(result, expected)

    def test_ner_parse_gpt_mini_2(self):
        pred = '''"```html
        <h1>Ketamine as a rapid treatment for <span class=""application-area"">chronic PTSD</span>: clinical effects and potential biological mechanisms.^</h1>
        <p>Background: Chronic stress and <span class=""application-area"">posttraumatic stress disorder (PTSD)</span> have been linked to glutamate system dysfunction. A sub‐anesthetic dose of ketamine, an N‐methyl‐D‐aspartate (NMDA) glutamate receptor antagonist, previously shown to be rapidly effective for treatment‐resistant depression, has been proposed as a potential <span class=""application-area"">PTSD</span> treatment. Methods: In a randomized, double‐blind, crossover trial, 41 patients with <span class=""application-area"">chronic PTSD</span>, assessed with the SCID‐P and CAPS, were assigned to receive one intravenous infusion of ketamine hydrochloride (<span class=""dosage"">0.5 mg/kg</span>) and one infusion of midazolam (<span class=""dosage"">0.045 mg/kg</span>), two weeks apart, in randomized counterbalanced order. The primary outcome was improvement on the Impact of Event Scale‐Revised (IES‐R) 24 hours post‐infusion. Additional measures included the CAPS, Montgomery‐Asberg Depression Rating Scale (MADRS), Clinical Global Severity Scales (CGI), among others. Results: <span class=""application-area"">PTSD</span> symptom improvement after ketamine infusion was rapid and significantly better than after midazolam, measured 24 hours post‐infusion (mean difference in IES‐R score = 12.7 [95% CI: 2.5‐22.8], p = .02, in analysis of crossover data). Analyses of <span class=""application-area"">PTSD</span> symptom improvement after the first infusion, and after adjusting for baseline and 24‐hour depressive symptoms, paralleled this finding. Ketamine infusion was also associated with improvement in comorbid depressive symptoms and GCI scores, with short‐lived, non‐clinically significant dissociative symptoms. Conclusions: Results provide the first evidence of rapid improvement in core <span class=""application-area"">PTSD</span> symptoms after ketamine infusion in <span class=""application-area"">chronic PTSD</span> patients. Implications for future pharmacologic approaches in patients with <span class=""application-area"">chronic PTSD</span> will be discussed, as well as potential biological mechanisms underlying the effects of ketamine in this disabling chronic condition.</p>
        ```"'''

        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        tokens = ['Ketamine', 'as', 'a', 'rapid', 'treatment', 'for', 'chronic', 'ptsd', ':', 'clinical', 'effects', 'and', 'potential', 'biological', 'mechanisms.^', '\n', 'Background', ':', 'Chronic', 'stress', 'and', 'posttraumatic', 'stress', 'disorder', '(', 'PTSD', ')', 'have', 'been', 'linked', 'to', 'glutamate', 'system', 'dysfunction', '.', 'A', 'sub‐anesthetic', 'dose', 'of', 'ketamine', ',', 'an', 'N‐methyl‐D‐aspartate', '(', 'NMDA', ')', 'glutamate', 'receptor', 'antagonist', ',', 'previously', 'shown', 'to', 'be', 'rapidly', 'effective', 'for', 'treatment‐resistant', 'depression', ',', 'has', 'been', 'proposed', 'as', 'a', 'potential', 'PTSD', 'treatment', '.', 'Methods', ':', 'In', 'a', 'randomized', ',', 'double‐blind', ',', 'crossover', 'trial', ',', '41', 'patients', 'with', 'chronic', 'PTSD', ',', 'assessed', 'with', 'the', 'SCID‐P', 'and', 'CAPS', ',', 'were', 'assigned', 'to', 'receive', 'one', 'intravenous', 'infusion', 'of', 'ketamine', 'hydrochloride', '(', '0.5', 'mg', '/', 'kg', ')', 'and', 'one', 'infusion', 'of', 'midazolam', '(', '0.045', 'mg', '/', 'kg', ')', ',', 'two', 'weeks', 'apart', ',', 'in', 'randomized', 'counterbalanced', 'order', '.', 'The', 'primary', 'outcome', 'was', 'improvement', 'on', 'the', 'Impact', 'of', 'Event', 'Scale‐Revised', '(', 'IES‐R', ')', '24', 'hours', 'post‐infusion', '.', 'Additional', 'measures', 'included', 'the', 'CAPS', ',', 'Montgomery‐Asberg', 'Depression', 'Rating', 'Scale', '(', 'MADRS', ')', ',',
                  'Clinical', 'Global', 'Severity', 'Scales', '(', 'CGI', ')', ',', 'among', 'others', '.', 'Results', ':', 'PTSD', 'symptom', 'improvement', 'after', 'ketamine', 'infusion', 'was', 'rapid', 'and', 'significantly', 'better', 'than', 'after', 'midazolam', ',', 'measured', '24', 'hours', 'post‐infusion', '(', 'mean', 'difference', 'in', 'IES‐R', 'score', '=', '12.7', '[', '95', '%', 'CI', ':', '2.5‐22.8', ']', ',', 'p', '=', '.02', ',', 'in', 'analysis', 'of', 'crossover', 'data', ')', '.', 'Analyses', 'of', 'PTSD', 'symptom', 'improvement', 'after', 'the', 'first', 'infusion', ',', 'and', 'after', 'adjusting', 'for', 'baseline', 'and', '24‐hour', 'depressive', 'symptoms', ',', 'paralleled', 'this', 'finding', '.', 'Ketamine', 'infusion', 'was', 'also', 'associated', 'with', 'improvement', 'in', 'comorbid', 'depressive', 'symptoms', 'and', 'GCI', 'scores', ',', 'with', 'short‐lived', ',', 'non‐clinically', 'significant', 'dissociative', 'symptoms', '.', 'Conclusions', ':', 'Results', 'provide', 'the', 'first', 'evidence', 'of', 'rapid', 'improvement', 'in', 'core', 'PTSD', 'symptoms', 'after', 'ketamine', 'infusion', 'in', 'chronic', 'PTSD', 'patients', '.', 'Implications', 'for', 'future', 'pharmacologic', 'approaches', 'in', 'patients', 'with', 'chronic', 'PTSD', 'will', 'be', 'discussed', ',', 'as', 'well', 'as', 'potential', 'biological', 'mechanisms', 'underlying', 'the', 'effects', 'of', 'ketamine', 'in', 'this', 'disabling', 'chronic', 'condition', '.']

        model = 'gpt-4o-mini-06-06-06'

        result = parse_ner_prediction(pred, tokens, model)
        # for r, e, t in zip(result, expected, tokens):
        #     print(r, e, t)
        self.assertEqual(result, expected)

    def test_ner_parse_llama2(self):
        model = 'ner_Llama-2-13-chat-hf'
        pred = '''"[INST] <<SYS>>
            You are a helpful medical expert who is helping to extract named entities from medical abstracts.
            <</SYS>>
            ###Task

            Your task is to generate an HTML version of an input text, marking up specific entities. The entities to be identified are: Application area, Dosage. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of entity.

            ###Entity Markup Guide
            Use <span class=""application-area""> to denote Application area, <span class=""dosage""> to denote Dosage.

            ###Entity Definitions
            Application area is defined as: The condition or disease studied in the paper, defined as a medically recognized state with identifiable diagnostic features, progression, and treatment response, typically having an ICD-10/11 code.

            Dosage is defined as: The applied dosage(s) of the psychedelic substance(s) in the present study.

            ###Annotation Guidelines
            Application area should be annotated according to the following criteria:
            * Can be empty, i.e., no annotation per abstract.
            * Annotate only in title and/or introductory abstract parts discussing the problem or objective (including abbreviations).
            * Annotate sweeping statements like ""major depressive and bipolar disorder"" as one entity.
            * Annotate full descriptive phrases (e.g., ""active suicidal ideation""); annotate adjectives like ""Depressed"" but not generic words like ""patients"".
            * Include defining features such as Acute/Chronic, Mild/Severe, Drug-resistant, Post-stroke, etc.
            * Annotate abbreviations separately, e.g., ""Huntington disease"" and ""HD"" in ""Huntington disease (HD)"".
            * Exclude study targets not treatment targets (e.g., ""dissociative symptoms"" in some contexts).
            * Annotate disorders like ""cigarette addiction"" but not concepts like ""smoking cessation"".
            * Exclude targets of study that are not treatment targets (e.g., ""dissociative symptoms"" in some contexts).
            * Do NOT annotate generic concepts like ""psychiatric condition"".
            * Do NOT annotate outcomes or measures like ""Hamilton Depression Scale"", ""antisuicidal response"", or phrases like ""depression was reduced"".
            * Do NOT annotate treatments like ""antidepressant"".

            Dosage should be annotated according to the following criteria:
            * Annotate the number and the corresponding unit (e.g., 2mg/kg).
            * Can be empty, i.e, no annotation per abstract.
            * Annotate only doses of the psychedelic substance in the current study (e.g., exclude doses mentioned from other studies).
            * Annotate all reported doses for the respective psychedelic substance.
            * Annotate also the term ""between"" if pertinent, similarly ""or"" and ""and"" should be annotated. Do not annotate ""per device"".
            * Annotate also relative dosages (e.g., 1 mg/kg), do NOT annotate the terms ""bw"" or ""body weight"" or similar if pertinent
            * Do NOT annotate concentrations of solutions (e.g., 500mg/10ml) or doses measured in plasma (4 mg/ml).

            INPUT: Resting-state Network-specific Breakdown of Functional Connectivity during Ketamine Alteration of Consciousness in Volunteers.^
            BACKGROUND: Consciousness-altering anesthetic agents disturb connectivity between brain regions composing the resting-state consciousness networks (RSNs). The default mode network (DMn), executive control network, salience network (SALn), auditory network, sensorimotor network (SMn), and visual network sustain mentation. Ketamine modifies consciousness differently from other agents, producing psychedelic dreaming and no apparent interaction with the environment. The authors used functional magnetic resonance imaging to explore ketamine-induced changes in RSNs connectivity. METHODS: Fourteen healthy volunteers received stepwise intravenous infusions of ketamine up to loss of responsiveness. Because of agitation, data from six subjects were excluded from analysis. RSNs connectivity was compared between absence of ketamine (wake state [W1]), light ketamine sedation, and ketamine-induced unresponsiveness (deep sedation [S2]). RESULTS: Increasing the depth of ketamine sedation from W1 to S2 altered DMn and SALn connectivity and suppressed the anticorrelated activity between DMn and other brain regions. During S2, DMn connectivity, particularly between the medial prefrontal cortex and the remaining network (effect size β [95% CI]: W1 = 0.20 [0.18 to 0.22]; S2 = 0.07 [0.04 to 0.09]), and DMn anticorrelated activity (e.g., right sensory cortex: W1 = -0.07 [-0.09 to -0.04]; S2 = 0.04 [0.01 to 0.06]) were broken down. SALn connectivity was nonuniformly suppressed (e.g., left parietal operculum: W1 = 0.08 [0.06 to 0.09]; S2 = 0.05 [0.02 to 0.07]). Executive control networks, auditory network, SMn, and visual network were minimally affected. CONCLUSIONS: Ketamine induces specific changes in connectivity within and between RSNs. Breakdown of frontoparietal DMn connectivity and DMn anticorrelation and sensory and SMn connectivity preservation are common to ketamine and propofol-induced alterations of consciousness.

            OUTPUT: [/INST]  Sure, I'd be happy to help! Here's the input text with the named entities annotated using HTML <span> tags and the appropriate class attributes:

            Resting-state Network-specific Breakdown of Functional Connectivity during Ketamine Alteration of Consciousness in Volunteers.^

            <span class=""application-area"">Consciousness-altering anesthetic agents</span> disturb connectivity between brain regions composing the resting-state consciousness networks (RSNs). The default mode network (DMn), executive control network, salience network (SALn), auditory network, sensorimotor network (SMn), and visual network sustain mentation. Ketamine modifies consciousness differently from other agents, producing psychedelic dreaming and no apparent interaction with the environment. The authors used functional magnetic resonance imaging to explore ketamine-induced changes in RSNs connectivity.

            <span class=""dosage"">Fourteen healthy volunteers received stepwise intravenous infusions of ketamine up to loss of responsiveness.</span> Because of agitation, data from six subjects were excluded from analysis. RSNs connectivity was compared between absence of ketamine (wake state [W1]), light ketamine sedation, and ketamine-induced unresponsiveness (deep sedation [S2]).

            <span class=""application-area"">Increasing the depth of ketamine sedation from W1 to S2 altered DMn and SALn connectivity and suppressed the anticorrelated activity between DMn and other brain regions.</span> During S2, DMn connectivity, particularly between the medial prefrontal cortex and the remaining network (effect size β [95% CI]: W1 = 0.20 [0.18 to 0.22]; S2 = 0.07 [0.04 to 0.09]), and DMn anticorrelated activity (e.g., right sensory cortex: W1 = -0.07 [-0.09 to -0.04]; S2 = 0.04 [0.01 to 0"'''

        tokens = ['Resting', '-', 'state', 'Network', '-', 'specific', 'Breakdown', 'of', 'Functional', 'Connectivity', 'during', 'Ketamine', 'Alteration', 'of', 'Consciousness', 'in', 'Volunteers.^', '\n', 'BACKGROUND', ':', 'Consciousness', '-', 'altering', 'anesthetic', 'agents', 'disturb', 'connectivity', 'between', 'brain', 'regions', 'composing', 'the', 'resting', '-', 'state', 'consciousness', 'networks', '(', 'RSNs', ')', '.', 'The', 'default', 'mode', 'network', '(', 'DMn', ')', ',', 'executive', 'control', 'network', ',', 'salience', 'network', '(', 'SALn', ')', ',', 'auditory', 'network', ',', 'sensorimotor', 'network', '(', 'SMn', ')', ',', 'and', 'visual', 'network', 'sustain', 'mentation', '.', 'Ketamine', 'modifies', 'consciousness', 'differently', 'from', 'other', 'agents', ',', 'producing', 'psychedelic', 'dreaming', 'and', 'no', 'apparent', 'interaction', 'with', 'the', 'environment', '.', 'The', 'authors', 'used', 'functional', 'magnetic', 'resonance', 'imaging', 'to', 'explore', 'ketamine', '-', 'induced', 'changes', 'in', 'RSNs', 'connectivity', '.', 'METHODS', ':', 'Fourteen', 'healthy', 'volunteers', 'received', 'stepwise', 'intravenous', 'infusions', 'of', 'ketamine', 'up', 'to', 'loss', 'of', 'responsiveness', '.', 'Because', 'of', 'agitation', ',', 'data', 'from', 'six', 'subjects', 'were', 'excluded', 'from', 'analysis', '.', 'RSNs', 'connectivity', 'was', 'compared', 'between', 'absence', 'of', 'ketamine', '(', 'wake', 'state', '[', 'W1', ']', ')', ',', 'light', 'ketamine', 'sedation', ',', 'and', 'ketamine', '-', 'induced', 'unresponsiveness', '(', 'deep', 'sedation', '[', 'S2', ']', ')',
                  '.', 'RESULTS', ':', 'Increasing', 'the', 'depth', 'of', 'ketamine', 'sedation', 'from', 'W1', 'to', 'S2', 'altered', 'DMn', 'and', 'SALn', 'connectivity', 'and', 'suppressed', 'the', 'anticorrelated', 'activity', 'between', 'DMn', 'and', 'other', 'brain', 'regions', '.', 'During', 'S2', ',', 'DMn', 'connectivity', ',', 'particularly', 'between', 'the', 'medial', 'prefrontal', 'cortex', 'and', 'the', 'remaining', 'network', '(', 'effect', 'size', 'β', '[', '95', '%', 'CI', ']', ':', 'W1', '=', '0.20', '[', '0.18', 'to', '0.22', ']', ';', 'S2', '=', '0.07', '[', '0.04', 'to', '0.09', ']', ')', ',', 'and', 'DMn', 'anticorrelated', 'activity', '(', 'e.g.', ',', 'right', 'sensory', 'cortex', ':', 'W1', '=', '-0.07', '[', '-0.09', 'to', '-0.04', ']', ';', 'S2', '=', '0.04', '[', '0.01', 'to', '0.06', ']', ')', 'were', 'broken', 'down', '.', 'SALn', 'connectivity', 'was', 'nonuniformly', 'suppressed', '(', 'e.g.', ',', 'left', 'parietal', 'operculum', ':', 'W1', '=', '0.08', '[', '0.06', 'to', '0.09', ']', ';', 'S2', '=', '0.05', '[', '0.02', 'to', '0.07', ']', ')', '.', 'Executive', 'control', 'networks', ',', 'auditory', 'network', ',', 'SMn', ',', 'and', 'visual', 'network', 'were', 'minimally', 'affected', '.', 'CONCLUSIONS', ':', 'Ketamine', 'induces', 'specific', 'changes', 'in', 'connectivity', 'within', 'and', 'between', 'RSNs', '.', 'Breakdown', 'of', 'frontoparietal', 'DMn', 'connectivity', 'and', 'DMn', 'anticorrelation', 'and', 'sensory', 'and', 'SMn', 'connectivity', 'preservation', 'are', 'common', 'to', 'ketamine', 'and', 'propofol', '-', 'induced', 'alterations', 'of', 'consciousness', '.']

        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area',
                    'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        result = parse_ner_prediction(pred, tokens, model)
        self.assertEqual(result, expected)

    def test_parse_ner_mellama(self):
        model = 'MeLLaMA-13B-chat'
        pred = '''"###Task
        Your task is to generate an HTML version of an input text, marking up specific entities. The entities to be identified are: Application area, Dosage. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of entity.

        ###Entity Markup Guide
        Use <span class=""application-area""> to denote Application area, <span class=""dosage""> to denote Dosage.

        ###Entity Definitions
        Application area is defined as: The condition or disease studied in the paper, defined as a medically recognized state with identifiable diagnostic features, progression, and treatment response, typically having an ICD-10/11 code.

        Dosage is defined as: The applied dosage(s) of the psychedelic substance(s) in the present study.

        ###Annotation Guidelines
        Application area should be annotated according to the following criteria:
        * Can be empty, i.e., no annotation per abstract.
        * Annotate only in title and/or introductory abstract parts discussing the problem or objective (including abbreviations).
        * Annotate sweeping statements like ""major depressive and bipolar disorder"" as one entity.
        * Annotate full descriptive phrases (e.g., ""active suicidal ideation""); annotate adjectives like ""Depressed"" but not generic words like ""patients"".
        * Include defining features such as Acute/Chronic, Mild/Severe, Drug-resistant, Post-stroke, etc.
        * Annotate abbreviations separately, e.g., ""Huntington disease"" and ""HD"" in ""Huntington disease (HD)"".
        * Exclude study targets not treatment targets (e.g., ""dissociative symptoms"" in some contexts).
        * Annotate disorders like ""cigarette addiction"" but not concepts like ""smoking cessation"".
        * Exclude targets of study that are not treatment targets (e.g., ""dissociative symptoms"" in some contexts).
        * Do NOT annotate generic concepts like ""psychiatric condition"".
        * Do NOT annotate outcomes or measures like ""Hamilton Depression Scale"", ""antisuicidal response"", or phrases like ""depression was reduced"".
        * Do NOT annotate treatments like ""antidepressant"".

        Dosage should be annotated according to the following criteria:
        * Annotate the number and the corresponding unit (e.g., 2mg/kg).
        * Can be empty, i.e, no annotation per abstract.
        * Annotate only doses of the psychedelic substance in the current study (e.g., exclude doses mentioned from other studies).
        * Annotate all reported doses for the respective psychedelic substance.
        * Annotate also the term ""between"" if pertinent, similarly ""or"" and ""and"" should be annotated. Do not annotate ""per device"".
        * Annotate also relative dosages (e.g., 1 mg/kg), do NOT annotate the terms ""bw"" or ""body weight"" or similar if pertinent
        * Do NOT annotate concentrations of solutions (e.g., 500mg/10ml) or doses measured in plasma (4 mg/ml).

        INPUT: Default Mode Connectivity in Major Depressive Disorder Measured Up to 10 Days After Ketamine Administration.^
        BACKGROUND: The symptoms of major depressive disorder (MDD) are rapidly alleviated by administration of a single dose of the glutamatergic modulator ketamine. However, few studies have investigated the potential sustained neural effects of this agent beyond immediate infusion. This study used functional magnetic resonance imaging to examine the effect of a single ketamine infusion on the resting state default mode network (DMN) at 2 and 10 days after a single ketamine infusion in unmedicated subjects with MDD as well as healthy control subjects (HCs). METHODS: Data were drawn from a double-blind, placebo-controlled crossover study of 58 participants (33 with MDD and 25 HCs) who received an intravenous infusion of either ketamine hydrochloride (0.5 mg/kg) or placebo on 2 separate test days spaced 2 weeks apart. Eight minutes of functional magnetic resonance imaging resting state data was acquired at baseline and at about 2 and 10 days after both infusions. The DMN was defined using seed-based correlation and was compared across groups and scans. RESULTS: In subjects with MDD, connectivity between the insula and the DMN was normalized compared with HCs 2 days postketamine infusion. This change was reversed after 10 days and did not appear in either of the placebo scans. Group-specific connectivity differences in drug response were observed, most notably in the insula in subjects with MDD and in the thalamus in HCs. CONCLUSIONS: Connectivity changes in the insula in subjects with MDD suggest that ketamine may normalize the interaction between the DMN and salience networks, supporting the triple network dysfunction model of MDD.

        OUTPUT: 
        <span class=""application-area""> Major Depressive Disorder </span> <span class=""dosage""> 0.5 mg/kg </span>".'''

        tokens = ['Default', 'Mode', 'Connectivity', 'in', 'Major', 'Depressive', 'Disorder', 'Measured', 'Up', 'to', '10', 'Days', 'After', 'Ketamine', 'Administration.^', '\n', 'BACKGROUND', ':', 'The', 'symptoms', 'of', 'major', 'depressive', 'disorder', '(', 'MDD', ')', 'are', 'rapidly', 'alleviated', 'by', 'administration', 'of', 'a', 'single', 'dose', 'of', 'the', 'glutamatergic', 'modulator', 'ketamine', '.', 'However', ',', 'few', 'studies', 'have', 'investigated', 'the', 'potential', 'sustained', 'neural', 'effects', 'of', 'this', 'agent', 'beyond', 'immediate', 'infusion', '.', 'This', 'study', 'used', 'functional', 'magnetic', 'resonance', 'imaging', 'to', 'examine', 'the', 'effect', 'of', 'a', 'single', 'ketamine', 'infusion', 'on', 'the', 'resting', 'state', 'default', 'mode', 'network', '(', 'DMN', ')', 'at', '2', 'and', '10', 'days', 'after', 'a', 'single', 'ketamine', 'infusion', 'in', 'unmedicated', 'subjects', 'with', 'MDD', 'as', 'well', 'as', 'healthy', 'control', 'subjects', '(', 'HCs', ')', '.', 'METHODS', ':', 'Data', 'were', 'drawn', 'from', 'a', 'double', '-', 'blind', ',', 'placebo', '-', 'controlled', 'crossover', 'study', 'of', '58', 'participants', '(', '33', 'with', 'MDD', 'and', '25', 'HCs', ')', 'who', 'received', 'an', 'intravenous', 'infusion', 'of', 'either', 'ketamine', 'hydrochloride', '(', '0.5', 'mg', '/', 'kg', ')', 'or',
                  'placebo', 'on', '2', 'separate', 'test', 'days', 'spaced', '2', 'weeks', 'apart', '.', 'Eight', 'minutes', 'of', 'functional', 'magnetic', 'resonance', 'imaging', 'resting', 'state', 'data', 'was', 'acquired', 'at', 'baseline', 'and', 'at', 'about', '2', 'and', '10', 'days', 'after', 'both', 'infusions', '.', 'The', 'DMN', 'was', 'defined', 'using', 'seed', '-', 'based', 'correlation', 'and', 'was', 'compared', 'across', 'groups', 'and', 'scans', '.', 'RESULTS', ':', 'In', 'subjects', 'with', 'MDD', ',', 'connectivity', 'between', 'the', 'insula', 'and', 'the', 'DMN', 'was', 'normalized', 'compared', 'with', 'HCs', '2', 'days', 'postketamine', 'infusion', '.', 'This', 'change', 'was', 'reversed', 'after', '10', 'days', 'and', 'did', 'not', 'appear', 'in', 'either', 'of', 'the', 'placebo', 'scans', '.', 'Group', '-', 'specific', 'connectivity', 'differences', 'in', 'drug', 'response', 'were', 'observed', ',', 'most', 'notably', 'in', 'the', 'insula', 'in', 'subjects', 'with', 'MDD', 'and', 'in', 'the', 'thalamus', 'in', 'HCs', '.', 'CONCLUSIONS', ':', 'Connectivity', 'changes', 'in', 'the', 'insula', 'in', 'subjects', 'with', 'MDD', 'suggest', 'that', 'ketamine', 'may', 'normalize', 'the', 'interaction', 'between', 'the', 'DMN', 'and', 'salience', 'networks', ',', 'supporting', 'the', 'triple', 'network', 'dysfunction', 'model', 'of', 'MDD', '.']

        expected = ['O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Dosage', 'I-Dosage',
                    'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        result = parse_ner_prediction(pred, tokens, model)
        self.assertEqual(result, expected)

    def test_ner_parse_mellama_2(self):
        model = 'MeLLaMA-13B-chat'
        pred = '''"###Task

            Your task is to generate an HTML version of an input text, marking up specific entities. The entities to be identified are: Application area, Dosage. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of entity.

            ###Entity Markup Guide
            Use <span class=""application-area""> to denote Application area, <span class=""dosage""> to denote Dosage.

            ###Entity Definitions
            Application area is defined as: The condition or disease studied in the paper, defined as a medically recognized state with identifiable diagnostic features, progression, and treatment response, typically having an ICD-10/11 code.

            Dosage is defined as: The applied dosage(s) of the psychedelic substance(s) in the present study.

            ###Annotation Guidelines
            Application area should be annotated according to the following criteria:
            * Can be empty, i.e., no annotation per abstract.
            * Annotate only in title and/or introductory abstract parts discussing the problem or objective (including abbreviations).
            * Annotate sweeping statements like ""major depressive and bipolar disorder"" as one entity.
            * Annotate full descriptive phrases (e.g., ""active suicidal ideation""); annotate adjectives like ""Depressed"" but not generic words like ""patients"".
            * Include defining features such as Acute/Chronic, Mild/Severe, Drug-resistant, Post-stroke, etc.
            * Annotate abbreviations separately, e.g., ""Huntington disease"" and ""HD"" in ""Huntington disease (HD)"".
            * Exclude study targets not treatment targets (e.g., ""dissociative symptoms"" in some contexts).
            * Annotate disorders like ""cigarette addiction"" but not concepts like ""smoking cessation"".
            * Exclude targets of study that are not treatment targets (e.g., ""dissociative symptoms"" in some contexts).
            * Do NOT annotate generic concepts like ""psychiatric condition"".
            * Do NOT annotate outcomes or measures like ""Hamilton Depression Scale"", ""antisuicidal response"", or phrases like ""depression was reduced"".
            * Do NOT annotate treatments like ""antidepressant"".

            Dosage should be annotated according to the following criteria:
            * Annotate the number and the corresponding unit (e.g., 2mg/kg).
            * Can be empty, i.e, no annotation per abstract.
            * Annotate only doses of the psychedelic substance in the current study (e.g., exclude doses mentioned from other studies).
            * Annotate all reported doses for the respective psychedelic substance.
            * Annotate also the term ""between"" if pertinent, similarly ""or"" and ""and"" should be annotated. Do not annotate ""per device"".
            * Annotate also relative dosages (e.g., 1 mg/kg), do NOT annotate the terms ""bw"" or ""body weight"" or similar if pertinent
            * Do NOT annotate concentrations of solutions (e.g., 500mg/10ml) or doses measured in plasma (4 mg/ml).

            INPUT: Resting-state Network-specific Breakdown of Functional Connectivity during Ketamine Alteration of Consciousness in Volunteers.^
            BACKGROUND: Consciousness-altering anesthetic agents disturb connectivity between brain regions composing the resting-state consciousness networks (RSNs). The default mode network (DMn), executive control network, salience network (SALn), auditory network, sensorimotor network (SMn), and visual network sustain mentation. Ketamine modifies consciousness differently from other agents, producing psychedelic dreaming and no apparent interaction with the environment. The authors used functional magnetic resonance imaging to explore ketamine-induced changes in RSNs connectivity. METHODS: Fourteen healthy volunteers received stepwise intravenous infusions of ketamine up to loss of responsiveness. Because of agitation, data from six subjects were excluded from analysis. RSNs connectivity was compared between absence of ketamine (wake state [W1]), light ketamine sedation, and ketamine-induced unresponsiveness (deep sedation [S2]). RESULTS: Increasing the depth of ketamine sedation from W1 to S2 altered DMn and SALn connectivity and suppressed the anticorrelated activity between DMn and other brain regions. During S2, DMn connectivity, particularly between the medial prefrontal cortex and the remaining network (effect size β [95% CI]: W1 = 0.20 [0.18 to 0.22]; S2 = 0.07 [0.04 to 0.09]), and DMn anticorrelated activity (e.g., right sensory cortex: W1 = -0.07 [-0.09 to -0.04]; S2 = 0.04 [0.01 to 0.06]) were broken down. SALn connectivity was nonuniformly suppressed (e.g., left parietal operculum: W1 = 0.08 [0.06 to 0.09]; S2 = 0.05 [0.02 to 0.07]). Executive control networks, auditory network, SMn, and visual network were minimally affected. CONCLUSIONS: Ketamine induces specific changes in connectivity within and between RSNs. Breakdown of frontoparietal DMn connectivity and DMn anticorrelation and sensory and SMn connectivity preservation are common to ketamine and propofol-induced alterations of consciousness.

            OUTPUT: 
            <span class=""application-area"">Consciousness-altering anesthetic agents disturb connectivity between brain regions composing the resting-state consciousness networks (RSNs). The default mode network (DMn), executive control network, salience network (SALn), auditory network, sensorimotor network (SMn), and visual network sustain mentation. Ketamine modifies consciousness differently from other agents, producing psychedelic dreaming and no apparent interaction with the environment. The authors used functional magnetic resonance imaging to explore ketamine-induced changes in RSNs connectivity.</span>"'''
        tokens = ['Resting', '-', 'state', 'Network', '-', 'specific', 'Breakdown', 'of', 'Functional', 'Connectivity', 'during', 'Ketamine', 'Alteration', 'of', 'Consciousness', 'in', 'Volunteers.^', '\n', 'BACKGROUND', ':', 'Consciousness', '-', 'altering', 'anesthetic', 'agents', 'disturb', 'connectivity', 'between', 'brain', 'regions', 'composing', 'the', 'resting', '-', 'state', 'consciousness', 'networks', '(', 'RSNs', ')', '.', 'The', 'default', 'mode', 'network', '(', 'DMn', ')', ',', 'executive', 'control', 'network', ',', 'salience', 'network', '(', 'SALn', ')', ',', 'auditory', 'network', ',', 'sensorimotor', 'network', '(', 'SMn', ')', ',', 'and', 'visual', 'network', 'sustain', 'mentation', '.', 'Ketamine', 'modifies', 'consciousness', 'differently', 'from', 'other', 'agents', ',', 'producing', 'psychedelic', 'dreaming', 'and', 'no', 'apparent', 'interaction', 'with', 'the', 'environment', '.', 'The', 'authors', 'used', 'functional', 'magnetic', 'resonance', 'imaging', 'to', 'explore', 'ketamine', '-', 'induced', 'changes', 'in', 'RSNs', 'connectivity', '.', 'METHODS', ':', 'Fourteen', 'healthy', 'volunteers', 'received', 'stepwise', 'intravenous', 'infusions', 'of', 'ketamine', 'up', 'to', 'loss', 'of', 'responsiveness', '.', 'Because', 'of', 'agitation', ',', 'data', 'from', 'six', 'subjects', 'were', 'excluded', 'from', 'analysis', '.', 'RSNs', 'connectivity', 'was', 'compared', 'between', 'absence', 'of', 'ketamine', '(', 'wake', 'state', '[', 'W1', ']', ')', ',', 'light', 'ketamine', 'sedation', ',', 'and', 'ketamine', '-', 'induced', 'unresponsiveness', '(', 'deep', 'sedation', '[', 'S2', ']', ')',
                  '.', 'RESULTS', ':', 'Increasing', 'the', 'depth', 'of', 'ketamine', 'sedation', 'from', 'W1', 'to', 'S2', 'altered', 'DMn', 'and', 'SALn', 'connectivity', 'and', 'suppressed', 'the', 'anticorrelated', 'activity', 'between', 'DMn', 'and', 'other', 'brain', 'regions', '.', 'During', 'S2', ',', 'DMn', 'connectivity', ',', 'particularly', 'between', 'the', 'medial', 'prefrontal', 'cortex', 'and', 'the', 'remaining', 'network', '(', 'effect', 'size', 'β', '[', '95', '%', 'CI', ']', ':', 'W1', '=', '0.20', '[', '0.18', 'to', '0.22', ']', ';', 'S2', '=', '0.07', '[', '0.04', 'to', '0.09', ']', ')', ',', 'and', 'DMn', 'anticorrelated', 'activity', '(', 'e.g.', ',', 'right', 'sensory', 'cortex', ':', 'W1', '=', '-0.07', '[', '-0.09', 'to', '-0.04', ']', ';', 'S2', '=', '0.04', '[', '0.01', 'to', '0.06', ']', ')', 'were', 'broken', 'down', '.', 'SALn', 'connectivity', 'was', 'nonuniformly', 'suppressed', '(', 'e.g.', ',', 'left', 'parietal', 'operculum', ':', 'W1', '=', '0.08', '[', '0.06', 'to', '0.09', ']', ';', 'S2', '=', '0.05', '[', '0.02', 'to', '0.07', ']', ')', '.', 'Executive', 'control', 'networks', ',', 'auditory', 'network', ',', 'SMn', ',', 'and', 'visual', 'network', 'were', 'minimally', 'affected', '.', 'CONCLUSIONS', ':', 'Ketamine', 'induces', 'specific', 'changes', 'in', 'connectivity', 'within', 'and', 'between', 'RSNs', '.', 'Breakdown', 'of', 'frontoparietal', 'DMn', 'connectivity', 'and', 'DMn', 'anticorrelation', 'and', 'sensory', 'and', 'SMn', 'connectivity', 'preservation', 'are', 'common', 'to', 'ketamine', 'and', 'propofol', '-', 'induced', 'alterations', 'of', 'consciousness', '.']

        expected = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area',
                    'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'I-Application area', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        result = parse_ner_prediction(pred, tokens, model)
        self.assertEqual(result, expected)
