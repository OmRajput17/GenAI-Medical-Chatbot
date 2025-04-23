# doctors_data.py

doctors_database = {
    "Cardiology": [
        {
            "name": "Dr. Sarah Chen",
            "specialty": "Cardiology",
            "conditions": ["heart disease", "hypertension", "heart attack", "angina", "arrhythmia", "heart failure"],
            "address": "123 Heart Center Drive, New York, NY 10001",
            "contact": "(212) 555-7890",
            "rating": 4.8
        },
        {
            "name": "Dr. James Wilson",
            "specialty": "Cardiology",
            "conditions": ["heart disease", "coronary artery disease", "heart valve problems", "heart failure"],
            "address": "456 Medical Plaza, Chicago, IL 60601",
            "contact": "(312) 555-1234",
            "rating": 4.7
        }
    ],
    "Neurology": [
        {
            "name": "Dr. Michelle Park",
            "specialty": "Neurology",
            "conditions": ["migraine", "epilepsy", "stroke", "multiple sclerosis", "parkinson's disease", "headache"],
            "address": "789 Brain Health Blvd, Boston, MA 02115",
            "contact": "(617) 555-2345",
            "rating": 4.9
        },
        {
            "name": "Dr. Robert Johnson",
            "specialty": "Neurology",
            "conditions": ["alzheimer's", "dementia", "seizures", "neuropathy", "headache", "brain tumor"],
            "address": "321 Neuroscience Way, San Francisco, CA 94107",
            "contact": "(415) 555-6789",
            "rating": 4.6
        }
    ],
    "Pulmonology": [
        {
            "name": "Dr. David Martinez",
            "specialty": "Pulmonology",
            "conditions": ["asthma", "copd", "pneumonia", "bronchitis", "respiratory infection", "covid-19", "lung disease"],
            "address": "567 Lung Institute Road, Denver, CO 80202",
            "contact": "(303) 555-4567",
            "rating": 4.8
        },
        {
            "name": "Dr. Lisa Thompson",
            "specialty": "Pulmonology",
            "conditions": ["tuberculosis", "lung cancer", "sleep apnea", "pulmonary fibrosis", "respiratory failure"],
            "address": "890 Breathing Center Lane, Seattle, WA 98101",
            "contact": "(206) 555-8901",
            "rating": 4.7
        }
    ],
    "Gastroenterology": [
        {
            "name": "Dr. Thomas Garcia",
            "specialty": "Gastroenterology",
            "conditions": ["gerd", "acid reflux", "ibs", "crohn's disease", "ulcerative colitis", "gallstones", "hepatitis"],
            "address": "432 Digestive Health Drive, Houston, TX 77002",
            "contact": "(713) 555-3456",
            "rating": 4.9
        },
        {
            "name": "Dr. Emily Brown",
            "specialty": "Gastroenterology",
            "conditions": ["stomach ulcer", "celiac disease", "liver disease", "pancreatitis", "appendicitis"],
            "address": "765 GI Center Blvd, Miami, FL 33130",
            "contact": "(305) 555-7890",
            "rating": 4.8
        }
    ],
    "Dermatology": [
        {
            "name": "Dr. Michael Lee",
            "specialty": "Dermatology",
            "conditions": ["eczema", "psoriasis", "acne", "rosacea", "skin cancer", "dermatitis", "rash"],
            "address": "543 Skin Care Avenue, Los Angeles, CA 90025",
            "contact": "(310) 555-2345",
            "rating": 4.7
        },
        {
            "name": "Dr. Jennifer Smith",
            "specialty": "Dermatology",
            "conditions": ["hives", "fungal infection", "warts", "shingles", "herpes", "skin infection"],
            "address": "876 Dermatology Plaza, Atlanta, GA 30303",
            "contact": "(404) 555-6789",
            "rating": 4.8
        }
    ],
    "Orthopedics": [
        {
            "name": "Dr. William Jones",
            "specialty": "Orthopedics",
            "conditions": ["arthritis", "fracture", "joint pain", "osteoporosis", "back pain", "sprain", "dislocation"],
            "address": "234 Bone & Joint Street, Philadelphia, PA 19103",
            "contact": "(215) 555-1234",
            "rating": 4.9
        },
        {
            "name": "Dr. Amanda Taylor",
            "specialty": "Orthopedics",
            "conditions": ["tendonitis", "carpal tunnel", "sciatica", "herniated disc", "scoliosis"],
            "address": "678 Orthopedic Center Way, Phoenix, AZ 85004",
            "contact": "(602) 555-5678",
            "rating": 4.7
        }
    ],
    "Ophthalmology": [
        {
            "name": "Dr. Richard Wong",
            "specialty": "Ophthalmology",
            "conditions": ["glaucoma", "cataracts", "macular degeneration", "conjunctivitis", "eye infection", "vision problems"],
            "address": "901 Vision Care Drive, Dallas, TX 75201",
            "contact": "(214) 555-9012",
            "rating": 4.8
        },
        {
            "name": "Dr. Sophia Miller",
            "specialty": "Ophthalmology",
            "conditions": ["dry eye", "retinal detachment", "diabetic retinopathy", "eye injury", "stye"],
            "address": "345 Eye Institute Blvd, Minneapolis, MN 55403",
            "contact": "(612) 555-3456",
            "rating": 4.7
        }
    ],
    "ENT": [
        {
            "name": "Dr. Daniel Harris",
            "specialty": "ENT (Ear, Nose, and Throat)",
            "conditions": ["sinusitis", "tonsillitis", "ear infection", "hearing loss", "vertigo", "strep throat"],
            "address": "789 ENT Specialists Street, Portland, OR 97205",
            "contact": "(503) 555-7890",
            "rating": 4.8
        },
        {
            "name": "Dr. Rachel White",
            "specialty": "ENT (Ear, Nose, and Throat)",
            "conditions": ["tinnitus", "nasal polyps", "deviated septum", "sleep apnea", "laryngitis"],
            "address": "432 Head & Neck Center Road, Detroit, MI 48226",
            "contact": "(313) 555-2345",
            "rating": 4.9
        }
    ],
    "Endocrinology": [
        {
            "name": "Dr. Katherine Robinson",
            "specialty": "Endocrinology",
            "conditions": ["diabetes", "thyroid disorders", "hormone imbalance", "hypothyroidism", "hyperthyroidism"],
            "address": "654 Endocrine Institute Lane, San Diego, CA 92101",
            "contact": "(619) 555-6789",
            "rating": 4.7
        },
        {
            "name": "Dr. Steven Clark",
            "specialty": "Endocrinology",
            "conditions": ["adrenal disorders", "pituitary disorders", "osteoporosis", "metabolic disorders", "diabetes"],
            "address": "987 Hormone Health Avenue, St. Louis, MO 63103",
            "contact": "(314) 555-9012",
            "rating": 4.8
        }
    ],
    "Psychiatry": [
        {
            "name": "Dr. Nicole Baker",
            "specialty": "Psychiatry",
            "conditions": ["depression", "anxiety", "bipolar disorder", "schizophrenia", "ptsd", "ocd"],
            "address": "321 Mental Health Plaza, Denver, CO 80204",
            "contact": "(303) 555-3456",
            "rating": 4.9
        },
        {
            "name": "Dr. Jason Martin",
            "specialty": "Psychiatry",
            "conditions": ["adhd", "eating disorders", "insomnia", "addiction", "panic attacks"],
            "address": "765 Behavioral Health Way, Austin, TX 78701",
            "contact": "(512) 555-7890",
            "rating": 4.8
        }
    ],
    "General Practice": [
        {
            "name": "Dr. Elizabeth Davis",
            "specialty": "General Practice",
            "conditions": ["common cold", "flu", "fever", "infection", "general checkup", "preventive care"],
            "address": "123 Family Medicine Drive, Nashville, TN 37203",
            "contact": "(615) 555-1234",
            "rating": 4.7
        },
        {
            "name": "Dr. Christopher Lewis",
            "specialty": "General Practice",
            "conditions": ["all general conditions", "referrals", "basic diagnostics", "wellness exams"],
            "address": "456 Primary Care Street, Pittsburgh, PA 15222",
            "contact": "(412) 555-5678",
            "rating": 4.8
        }
    ]
}

# Dictionary to map conditions to specialties
condition_to_specialty = {}
for specialty, doctors_list in doctors_database.items():
    for doctor in doctors_list:
        for condition in doctor["conditions"]:
            condition_to_specialty[condition.lower()] = specialty

def find_doctors_for_conditions(conditions_list):
    """Find relevant doctors based on a list of potential conditions"""
    recommended_doctors = {}
    matched_specialties = set()
    
    # Process each condition to find matching specialties
    for condition in conditions_list:
        condition_lower = condition.lower().strip()
        
        # Check for exact matches
        for known_condition, specialty in condition_to_specialty.items():
            if condition_lower == known_condition or condition_lower in known_condition or known_condition in condition_lower:
                matched_specialties.add(specialty)
                break
    
    # If no matches found, add general practice as default
    if not matched_specialties:
        matched_specialties.add("General Practice")
    
    # Add doctors from matched specialties
    for specialty in matched_specialties:
        recommended_doctors[specialty] = doctors_database.get(specialty, [])
    
    return recommended_doctors