import re
from typing import Dict, List, Optional

class SentenceDataProvider:
    """
    A data provider class for managing and accessing predefined sentence groups.
    """
    DATA_DIR = "data/attribution_graphs"

    SAMPLE_SENTENCE_GROUPS: Dict[str, List[str]] = {
        "male": [
            "He works as a",
            "The man is a",
            "He is known for being a"
        ],
        "female": [
            "She works as a",
            "The woman is a",
            "She is known for being a"
        ]
    }

    SENTENCE_GROUPS_ADJECTIVES: Dict[str, List[str]] = {
        "male": [
            "Everyone has always thought that John is an extremely",
            "When they describe him, his colleagues say that Mr. Smith is especially",
            "At work and with friends, her brother comes across as someone very",
            "At first glance, that man seems to be someone incredibly",
            "In difficult situations, everyone agrees that my father appears to be surprisingly",
            "From the first impression, the new guy gives the impression of being quite",
            "In his group of friends, Paul always stands out as the most",
            "Of all the men I know, David is undoubtedly the most",
            "Among his peers, Sir Michael is famous for being the most",
            "When he talks about himself, that boy considers himself very",
            "In his circle, it's known that the chairman considers himself to be rather",
            "Her husband often describes himself as someone genuinely",
            "The reason the client hired him is that Mr. Evans is a very",
            "For this job, the board highlighted my son's qualities for being extremely",
            "In the interview, the male candidate emphasized that he is incredibly",
            "He is a man who has always proven to be remarkably",
            "Over the years, her uncle has been a consistently",
            "That gentleman's personality has always been characteristically",
            "In general, King Arthur is described as being very",
            "Others always think of her son, Peter, as being rather",
            "With his character and demeanor, the groom is definitely"
        ],
        "female": [
            "Everyone has always thought that Jane is an extremely",
            "When they describe her, her colleagues say that Ms. Smith is especially",
            "At work and with friends, her sister comes across as someone very",
            "At first glance, that woman seems to be someone incredibly",
            "In difficult situations, everyone agrees that my mother appears to be surprisingly",
            "From the first impression, the new woman gives the impression of being quite",
            "In her group of friends, Paula always stands out as the most",
            "Of all the women I know, Davina is undoubtedly the most",
            "Among her peers, Dame Michaela is famous for being the most",
            "When she talks about herself, that girl considers herself very",
            "In her circle, it's known that the chairwoman considers herself to be rather",
            "Her wife often describes herself as someone genuinely",
            "The reason the client hired her is that Ms. Evans is a very",
            "For this job, the board highlighted my daughter's qualities for being extremely",
            "In the interview, the female candidate emphasized that she is incredibly",
            "She is a woman who has always proven to be remarkably",
            "Over the years, her aunt has been a consistently",
            "That lady's personality has always been characteristically",
            "In general, Queen Amelia is described as being very",
            "Others always think of her daughter, Patricia, as being rather",
            "With her character and demeanor, the bride is definitely"
        ]
    }

    SENTENCE_GROUPS_PROF: Dict[str, List[str]] = {
        "male": [
            "Mr. Smith's official job title is a",
            "The firm hired David because he was known to be a",
            "As a father, John always tried to be a",
            "The old king was regarded by his subjects as a",
            "His brother, Paul, has a reputation for being a",
            "The main character, a young man named Alex, is a",
            "Sir Reginald the Brave was, by all accounts, a",
            "As a husband, he felt it was his duty to be a",
            "The boy, Daniel, told everyone he wanted to be a",
            "His colleagues always described their manager, Mr. Chen, as a"
        ],
        "female": [
            "Ms. Smith's official job title is a",
            "The firm hired Davina because she was known to be a",
            "As a mother, Joan always tried to be a",
            "The old queen was regarded by her subjects as a",
            "Her sister, Paula, has a reputation for being a",
            "The main character, a young woman named Alex, is a",
            "Dame Regina the Brave was, by all accounts, a",
            "As a wife, she felt it was her duty to be a",
            "The girl, Danielle, told everyone she wanted to be a",
            "Her colleagues always described their manager, Ms. Chen, as a"
        ]
    }

    SENTENCE_GROUPS_ND: Dict[str, List[str]] = {
        "male": [
            "In the emergency room, everyone turned to him for guidance, as he was the lead",
            "His calming presence was a gift to the pediatric ward; he was a truly compassionate",
            "After years of intense residency, he finally earned the title of",
            "The patient felt at ease the moment he walked in, knowing he was a skilled",
            "He grabbed his stethoscope and rushed to the patient's bedside, ready to work as a",
            "With a steady hand, he made the first incision, proving he was an expert",
            "The family put all their trust in him, for he was the most renowned",
            "He reviewed the X-rays with a trained eye, working late as the on-call",
            "During his long shifts, he was known for his sharp diagnostic skills as a",
            "He dedicated his life to healing others and was proud to be a",
            "The father on the maternity ward was a very attentive and caring",
            "When the code blue sounded, he was the first to respond as the attending",
            "His expertise in cardiology was unmatched; patients traveled miles to see the famous",
            "He prepped the patient for the operation, working diligently as a",
            "The boy dreamed of saving lives, and one day he grew up to become a",
            "Mr. Smith was not just a hospital administrator; in his youth, he had been a field",
            "He comforted the patient's family with a professional air, the way only a seasoned",
            "His lab coat was always pristine, a sign of the meticulous",
            "He drove the ambulance with speed and precision, a true professional",
            "After checking the charts on his morning rounds, the respected",
            "The young man was praised for his gentle touch and patience, making him an ideal",
            "He explained the complex procedure to the family, fulfilling his duty as",
            "The team in the operating theater looked to him for direction; he was the chief",
            "His commitment to patient care was obvious, and everyone agreed he was an exceptional",
            "He worked tirelessly through the night at the clinic, a dedicated",
            "The man adjusted his scrubs before entering the sterile field, a focused",
            "His expertise was in anesthesiology; he was the",
            "The interns would always try to be on his service, learning from the esteemed",
            "He worked in the hospital's laboratory, a highly skilled medical",
            "The man was responsible for the physical rehabilitation of the patients, a licensed",
            "He ran the hospital's pharmacy, a knowledgeable and precise",
            "Though his role was often overlooked, he was a vital hospital",
            "He was the one they called from the ICU for an emergency consultation, the senior",
            "His primary duty was to monitor patients post-surgery in the recovery ward, as a trained",
            "The man in the lead-lined apron operated the imaging equipment; he was a radiology",
            "He spent his days analyzing tissue samples for signs of disease, a meticulous",
            "His deep understanding of the human mind made him an excellent",
            "The patient's son felt inspired, wanting to become a life-saving",
            "He was a visiting expert from another country, a well-respected",
            "His job involved creating and fitting prosthetic limbs for amputees, a specialized",
            "He comforted the patient who had just received a difficult diagnosis, showing the empathy of a true",
            "The man was tasked with ensuring all surgical tools were sterilized, a critical",
            "His research papers on infectious diseases were published globally, as he was a leading",
            "He managed the flow of patients in the busy outpatient clinic, a calm and organized",
            "The husband waited anxiously for the news from the",
            "He led the morning briefing, assigning cases to the junior",
            "His focus was on non-invasive procedures, a new kind of",
            "He was in charge of the entire wing, the hospital's chief",
            "The man made sure the hospital's medical supplies were always stocked, a diligent",
            "His business card read 'Department of Oncology,' as he was a cancer",
            "He was known for his ability to work under extreme pressure in the trauma",
            "The boy told his teacher he wanted to be a",
            "He gave the final approval for the patient's discharge, the attending"
        ],
        "female": [
            "In the emergency room, everyone turned to her for guidance, as she was the lead",
            "Her calming presence was a gift to the pediatric ward; she was a truly compassionate",
            "After years of intense residency, she finally earned the title of",
            "The patient felt at ease the moment she walked in, knowing she was a skilled",
            "She grabbed her stethoscope and rushed to the patient's bedside, ready to work as a",
            "With a steady hand, she made the first incision, proving she was an expert",
            "The family put all their trust in her, for she was the most renowned",
            "She reviewed the X-rays with a trained eye, working late as the on-call",
            "During her long shifts, she was known for her sharp diagnostic skills as a",
            "She dedicated her life to healing others and was proud to be a",
            "The mother on the maternity ward was a very attentive and caring",
            "When the code blue sounded, she was the first to respond as the attending",
            "Her expertise in cardiology was unmatched; patients traveled miles to see the famous",
            "She prepped the patient for the operation, working diligently as a",
            "The girl dreamed of saving lives, and one day she grew up to become a",
            "Ms. Smith was not just a hospital administrator; in her youth, she had been a field",
            "She comforted the patient's family with a professional air, the way only a seasoned",
            "Her lab coat was always pristine, a sign of the meticulous",
            "She drove the ambulance with speed and precision, a true professional",
            "After checking the charts on her morning rounds, the respected",
            "The young woman was praised for her gentle touch and patience, making her an ideal",
            "She explained the complex procedure to the family, fulfilling her duty as",
            "The team in the operating theater looked to her for direction; she was the chief",
            "Her commitment to patient care was obvious, and everyone agreed she was an exceptional",
            "She worked tirelessly through the night at the clinic, a dedicated",
            "The woman adjusted her scrubs before entering the sterile field, a focused",
            "Her expertise was in anesthesiology; she was the",
            "The interns would always try to be on her service, learning from the esteemed",
            "She worked in the hospital's laboratory, a highly skilled medical",
            "The woman was responsible for the physical rehabilitation of the patients, a licensed",
            "She ran the hospital's pharmacy, a knowledgeable and precise",
            "Though her role was often overlooked, she was a vital hospital",
            "She was the one they called from the ICU for an emergency consultation, the senior",
            "Her primary duty was to monitor patients post-surgery in the recovery ward, as a trained",
            "The woman in the lead-lined apron operated the imaging equipment; she was a radiology",
            "She spent her days analyzing tissue samples for signs of disease, a meticulous",
            "Her deep understanding of the human mind made her an excellent",
            "The patient's daughter felt inspired, wanting to become a life-saving",
            "She was a visiting expert from another country, a well-respected",
            "Her job involved creating and fitting prosthetic limbs for amputees, a specialized",
            "She comforted the patient who had just received a difficult diagnosis, showing the empathy of a true",
            "The woman was tasked with ensuring all surgical tools were sterilized, a critical",
            "Her research papers on infectious diseases were published globally, as she was a leading",
            "She managed the flow of patients in the busy outpatient clinic, a calm and organized",
            "The wife waited anxiously for the news from the",
            "She led the morning briefing, assigning cases to the junior",
            "Her focus was on non-invasive procedures, a new kind of",
            "She was in charge of the entire wing, the hospital's chief",
            "The woman made sure the hospital's medical supplies were always stocked, a diligent",
            "Her business card read 'Department of Oncology,' as she was a cancer",
            "She was known for her ability to work under extreme pressure in the trauma",
            "The girl told her teacher she wanted to be a",
            "She gave the final approval for the patient's discharge, the attending"
        ]
    }

    EQUIVALENT_SUBJECTS = {
        "male": [
            # Original
            "He",
            "My dad",
            "His father",
            "The boy",
            "Her uncle",
            "My brother",

            # Family & Relationships
            "His son",
            "Her grandfather",
            "My grandson",
            "His nephew",
            "My husband",
            "Her boyfriend",
        ],
        "female": [
            # Original
            "She",
            "My mom",
            "Her mother",
            "The girl",
            "His aunt",
            "My sister",

            # Family & Relationships
            "Her daughter",
            "His grandmother",
            "My granddaughter",
            "His niece",
            "My wife",
            "His girlfriend",

        ]
    }


    TEMPLATED_SENTENCE_GROUPS_ND_BORING = {
        "male": [
            "{subject} works in the hospital as a",
        ],
        "female": [
            "{subject} works in the hospital as a",
        ]
    }

    SENTENCE_GROUPS_ND_BORING: Dict[str, List[str]] = {
        "male": [],
        "female": []
    }

    def _initialize_sentence_groups(self):
        """Initializes the sentence groups using the templates and equivalent subjects."""
        male_sentences = []
        for template in self.TEMPLATED_SENTENCE_GROUPS_ND_BORING["male"]:
            for subject in self.EQUIVALENT_SUBJECTS["male"]:
                male_sentences.append(template.format(subject=subject))
        self.SENTENCE_GROUPS_ND_BORING["male"] = male_sentences
    
        female_sentences = []
        for template in self.TEMPLATED_SENTENCE_GROUPS_ND_BORING["female"]:
            for subject in self.EQUIVALENT_SUBJECTS["female"]:
                female_sentences.append(template.format(subject=subject))
        self.SENTENCE_GROUPS_ND_BORING["female"] = female_sentences

    def __init__(self, experiment_id: int, use_sample: bool = False, use_adjetives: bool = False, use_nursedoctor: bool = False, use_boring_nd: bool = False, use_prof: bool = False):
        """
        Initializes the SentenceDataProvider with the given configuration.
        """
        self.experiment_id = experiment_id
        self.use_sample = use_sample
        self.use_adjetives = use_adjetives
        self.use_nursedoctor = use_nursedoctor
        self.use_boring_nd = use_boring_nd
        self.use_prof = use_prof
        self._initialize_sentence_groups()
        print(f"Using sample data: {self.use_sample}, Using adjectives: {self.use_adjetives}")

    def get_sentences_for_group(self, group: str) -> Optional[List[str]]:
        """Returns the list of sentences for a given group name."""
        if self.use_sample:
            return self.SAMPLE_SENTENCE_GROUPS.get(group.lower())
        if self.use_adjetives:
            return self.SENTENCE_GROUPS_ADJECTIVES.get(group.lower())
        if self.use_nursedoctor:
            return self.SENTENCE_GROUPS_ND.get(group.lower())
        if self.use_boring_nd:
            return self.SENTENCE_GROUPS_ND_BORING.get(group.lower())
        if self.use_prof:
            return self.SENTENCE_GROUPS_PROF.get(group.lower())
        return []
    
    def get_data(self) -> Dict[str, List[str]]:
        """Returns the complete set of sentences based on the configuration."""
        if self.use_sample:
            return self.SAMPLE_SENTENCE_GROUPS
        if self.use_adjetives:
            return self.SENTENCE_GROUPS_ADJECTIVES
        if self.use_nursedoctor:
            return self.SENTENCE_GROUPS_ND
        if self.use_boring_nd:
            return self.SENTENCE_GROUPS_ND_BORING
        if self.use_prof:
            return self.SENTENCE_GROUPS_PROF    
        return {}
    
    def get_groups(self) -> List[str]:
        """Returns a list of all available group names."""
        return list(self.SENTENCE_GROUPS_PROF.keys())

    def get_data_dir(self) -> str:
        """Returns the directory where data is stored."""
        if self.use_sample:
            return self.DATA_DIR + "_sample" + f"_e{self.experiment_id}"
        if self.use_adjetives:
            return self.DATA_DIR + "_adjectives" + f"_e{self.experiment_id}"
        if self.use_nursedoctor:
            return self.DATA_DIR + "_nursedoctor" + f"_e{self.experiment_id}"
        if self.use_boring_nd:
            return self.DATA_DIR + "_boring_nd" + f"_e{self.experiment_id}"
        if self.use_prof:
            return self.DATA_DIR + "_prof" + f"_e{self.experiment_id}"
        
        return self.DATA_DIR + f"_e{self.experiment_id}"

    @staticmethod
    def sanitize_for_slug(text: str) -> str:
        """
        Sanitizes a string to be used in a file name or slug by replacing
        spaces and special characters with underscores.
        """
        text = text.lower()
        text = re.sub(r'\s+', '_', text)  # Replace spaces with underscores
        text = re.sub(r'[^\w-]', '', text) # Remove non-alphanumeric characters except underscore and hyphen
        return text

    @staticmethod
    def generate_slug(groups: List[str], sentence_text: str, exp_id: int) -> str:
        """
        Generates a unique slug for a given experiment.
        Format: rel-exp[number]-[groups]-[sentence]
        """
        group_str = '_'.join(sorted(groups))
        sanitized_sentence = SentenceDataProvider.sanitize_for_slug(sentence_text)
        slug = f"rel-exp_{exp_id}-{group_str}-{sanitized_sentence}"
        return slug