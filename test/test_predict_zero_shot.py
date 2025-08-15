import unittest
from zero_shot.predict_zero_shot import parse_class_prediction, get_label2int


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
        input_data= '"{\n    ""Randomized-controlled trial (RCT)"": ""1"",\n    ""Cohort study"": ""0"",\n    ""Real-world study"": ""0"",\n    ""Study protocol"": ""0"",\n    ""Systematic review/meta-analysis"": ""0"",\n    ""Qualitative Study"": ""0"",\n    ""Case report"": ""0"",\n    ""Case series"": ""0"",\n    ""Other"": ""0""\n}'
        excepted = "[1, 0, 0, 0, 0, 0, 0, 0, 0]"
        results = parse_class_prediction(input_data, self.label2int, model)

    def test_parse_class_gpt_faulty(self):
        model = 'gpt-4o-mini-2024-07-18'
        input_data= '''"{
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
        input_data= '{\\n    "Randomized-controlled trial (RCT)": ",\\n    "Cohort study": ",\\n    "Real-world study": ",\\n    "Study protocol": ",\\n    "Systematic review/meta-analysis": "1",\\n    "Qualitative Study": ",\\n    "Case report": ",\\n    "Case series": ",\\n    "Other": "\\n}'
        excepted = "[0, 0, 0, 0, 1, 0, 0, 0, 0]"
        results = parse_class_prediction(input_data, self.label2int, model)
        self.assertEqual(excepted, results)