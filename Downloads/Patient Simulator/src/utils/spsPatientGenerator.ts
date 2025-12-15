// Standardized Patient Simulator (SPS) - Patient Case Generator
// Generates realistic patient cases with full clinical details

export interface SPSPatientCase {
  // Case metadata
  caseId: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  setting: 'clinic' | 'emergency_department';
  timeConstraint: string;
  
  // Patient demographics
  name: string;
  age: number;
  gender: string;
  pronouns: string;
  ethnicity: string;
  occupation: string;
  languageLevel: 'fluent' | 'moderate' | 'limited' | 'interpreter_needed';
  
  // Chief complaint
  chiefComplaint: string;
  
  // OLDCARTS details
  onset: string;
  location: string;
  duration: string;
  character: string;
  aggravatingFactors: string[];
  relievingFactors: string[];
  timing: string;
  severity: string;
  
  // Vitals
  vitals: {
    bloodPressure: string;
    heartRate: number;
    respiratoryRate: number;
    temperature: number;
    oxygenSaturation: number;
    painScale?: number;
  };
  
  // Current medications
  medications: string[];
  
  // Allergies
  allergies: string[];
  
  // Past medical history
  pastMedicalHistory: string[];
  
  // Past surgical history
  pastSurgicalHistory: string[];
  
  // Family history
  familyHistory: string;
  
  // Social history
  socialHistory: {
    smoking: string;
    alcohol: string;
    drugs: string;
    occupation: string;
    living: string;
  };
  
  // Sexual history (if relevant)
  sexualHistory?: string;
  
  // Red flags to identify
  redFlags: string[];
  
  // Distractors (misleading info)
  distractors: string[];
  
  // Physical exam findings
  physicalExamFindings: {
    general: string;
    [key: string]: string;
  };
  
  // Hidden diagnosis (ground truth)
  diagnosis: string;
  acceptedDifferentials: string[];
  
  // Patient personality traits
  personalityTraits: string[];
  emotionalState: string;
}

const firstNames = {
  male: ['James', 'Michael', 'Robert', 'John', 'David', 'William', 'Richard', 'Thomas', 'Christopher', 'Daniel'],
  female: ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy'],
  neutral: ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Avery', 'Cameron', 'Quinn', 'Sage']
};

const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee'];

const ethnicities = ['White British', 'Black British', 'Asian British', 'Mixed', 'Chinese', 'Other'];

const occupations = ['teacher', 'retail worker', 'office administrator', 'nurse', 'builder', 'software developer', 'chef', 'driver', 'unemployed', 'retired', 'student', 'accountant', 'engineer', 'cleaner', 'sales assistant'];

// Clinical case templates based on common NHS conditions
const clinicalCases = [
  {
    condition: 'Acute Appendicitis',
    difficulty: 'intermediate' as const,
    chiefComplaint: 'severe abdominal pain',
    onset: 'Started 12 hours ago',
    location: 'Started around umbilicus, now right lower quadrant',
    duration: '12 hours, getting worse',
    character: 'Sharp, constant pain',
    aggravatingFactors: ['movement', 'coughing', 'eating'],
    relievingFactors: ['lying still'],
    timing: 'Continuous, worse over last 6 hours',
    severity: '8/10',
    vitals: {
      bloodPressure: '128/82',
      heartRate: 102,
      respiratoryRate: 18,
      temperature: 38.2,
      oxygenSaturation: 98,
      painScale: 8
    },
    redFlags: ['fever', 'peritoneal signs', 'worsening pain', 'nausea and vomiting'],
    distractors: ['recent dietary changes', 'stress at work'],
    physicalExamFindings: {
      general: 'Appears uncomfortable, lying still',
      abdomen: 'Tender RLQ, positive McBurney\'s point, guarding present, rebound tenderness',
      skin: 'Warm, slightly flushed'
    },
    diagnosis: 'Acute appendicitis',
    differentials: ['Gastroenteritis', 'Ovarian cyst (if female)', 'Mesenteric adenitis', 'Urinary tract infection']
  },
  {
    condition: 'Type 2 Diabetes - New Diagnosis',
    difficulty: 'beginner' as const,
    chiefComplaint: 'increased thirst and urination',
    onset: 'Noticed over the past 3 months',
    location: 'Generalized',
    duration: '3 months, gradually worsening',
    character: 'Constant thirst, frequent urination',
    aggravatingFactors: ['any fluid intake leads to more urination'],
    relievingFactors: ['none identified'],
    timing: 'Throughout the day and night',
    severity: 'Interfering with sleep',
    vitals: {
      bloodPressure: '142/88',
      heartRate: 76,
      respiratoryRate: 14,
      temperature: 36.8,
      oxygenSaturation: 99
    },
    redFlags: ['unexplained weight loss', 'blurred vision', 'family history of diabetes'],
    distractors: ['recent stress', 'drinking more coffee'],
    physicalExamFindings: {
      general: 'Overweight (BMI 32)',
      skin: 'Dry, no lesions',
      cardiovascular: 'Regular rhythm, no murmurs',
      extremities: 'Peripheral pulses present bilaterally'
    },
    diagnosis: 'Type 2 Diabetes Mellitus',
    differentials: ['Diabetes insipidus', 'Hyperthyroidism', 'Psychogenic polydipsia']
  },
  {
    condition: 'Pneumonia',
    difficulty: 'intermediate' as const,
    chiefComplaint: 'cough with fever and shortness of breath',
    onset: 'Started 4 days ago',
    location: 'Chest, worse on right side',
    duration: '4 days, progressively worse',
    character: 'Productive cough with yellow-green sputum, fever, difficulty breathing',
    aggravatingFactors: ['deep breathing', 'lying flat', 'physical activity'],
    relievingFactors: ['sitting upright', 'paracetamol for fever'],
    timing: 'Constant, worse at night',
    severity: 'Breathing difficulty 7/10',
    vitals: {
      bloodPressure: '118/72',
      heartRate: 108,
      respiratoryRate: 24,
      temperature: 38.8,
      oxygenSaturation: 92
    },
    redFlags: ['hypoxia', 'tachycardia', 'tachypnea', 'high fever'],
    distractors: ['had a cold last week', 'works in an office'],
    physicalExamFindings: {
      general: 'Appears unwell, slightly dyspneic',
      respiratory: 'Decreased breath sounds right base, crackles present, dull to percussion',
      cardiovascular: 'Tachycardic, regular rhythm'
    },
    diagnosis: 'Community-acquired pneumonia',
    differentials: ['Bronchitis', 'Pulmonary embolism', 'Heart failure', 'Lung abscess']
  },
  {
    condition: 'Migraine',
    difficulty: 'beginner' as const,
    chiefComplaint: 'severe headache with nausea',
    onset: 'This morning, woke up with it',
    location: 'Left temple, behind left eye',
    duration: '6 hours so far',
    character: 'Throbbing, pulsating pain',
    aggravatingFactors: ['light', 'noise', 'movement'],
    relievingFactors: ['dark quiet room', 'tried paracetamol with minimal relief'],
    timing: 'Continuous since onset',
    severity: '9/10',
    vitals: {
      bloodPressure: '132/78',
      heartRate: 68,
      respiratoryRate: 14,
      temperature: 36.9,
      oxygenSaturation: 99,
      painScale: 9
    },
    redFlags: ['visual aura preceded headache', 'nausea and vomiting'],
    distractors: ['stressful week at work', 'missed breakfast'],
    physicalExamFindings: {
      general: 'Photophobic, appears in pain',
      neurological: 'Cranial nerves intact, no focal deficits, normal fundoscopy',
      neck: 'Supple, no meningismus'
    },
    diagnosis: 'Migraine with aura',
    differentials: ['Tension headache', 'Cluster headache', 'Subarachnoid hemorrhage', 'Meningitis']
  },
  {
    condition: 'Acute Myocardial Infarction',
    difficulty: 'advanced' as const,
    chiefComplaint: 'crushing chest pain',
    onset: '45 minutes ago, came on while resting',
    location: 'Central chest, radiating to left arm and jaw',
    duration: '45 minutes, constant',
    character: 'Heavy pressure, crushing sensation',
    aggravatingFactors: ['anxiety about the pain'],
    relievingFactors: ['nothing helps'],
    timing: 'Continuous since onset',
    severity: '10/10',
    vitals: {
      bloodPressure: '158/96',
      heartRate: 112,
      respiratoryRate: 22,
      temperature: 36.7,
      oxygenSaturation: 94,
      painScale: 10
    },
    redFlags: ['chest pain >20 minutes', 'radiation to arm/jaw', 'sweating', 'nausea', 'shortness of breath', 'risk factors present'],
    distractors: ['had indigestion last night', 'heavy meal earlier'],
    physicalExamFindings: {
      general: 'Anxious, diaphoretic, pale',
      cardiovascular: 'Tachycardic, regular rhythm, S3 gallop',
      respiratory: 'Mild crackles at bases bilaterally'
    },
    diagnosis: 'ST-elevation myocardial infarction (STEMI)',
    differentials: ['Unstable angina', 'Aortic dissection', 'Pulmonary embolism', 'Pericarditis', 'GERD']
  },
  {
    condition: 'Urinary Tract Infection',
    difficulty: 'beginner' as const,
    chiefComplaint: 'burning when urinating',
    onset: '2 days ago',
    location: 'Suprapubic discomfort, burning on urination',
    duration: '2 days, getting worse',
    character: 'Burning sensation, frequent urge to urinate',
    aggravatingFactors: ['drinking fluids increases frequency'],
    relievingFactors: ['none found'],
    timing: 'With each urination, 10-12 times per day',
    severity: '6/10 discomfort',
    vitals: {
      bloodPressure: '118/74',
      heartRate: 78,
      respiratoryRate: 14,
      temperature: 37.2,
      oxygenSaturation: 99
    },
    redFlags: ['blood in urine', 'suprapubic pain'],
    distractors: ['recently started new soap', 'sexually active'],
    physicalExamFindings: {
      general: 'Comfortable at rest',
      abdomen: 'Suprapubic tenderness, no flank pain',
      external: 'No lesions or discharge noted'
    },
    diagnosis: 'Uncomplicated urinary tract infection',
    differentials: ['Pyelonephritis', 'Sexually transmitted infection', 'Interstitial cystitis', 'Urethritis']
  }
];

export function generateSPSPatient(): SPSPatientCase {
  // Select random case template
  const caseTemplate = clinicalCases[Math.floor(Math.random() * clinicalCases.length)];
  
  // Generate demographics
  const gender = Math.random() > 0.5 ? 'male' : 'female';
  const firstName = gender === 'male' 
    ? firstNames.male[Math.floor(Math.random() * firstNames.male.length)]
    : firstNames.female[Math.floor(Math.random() * firstNames.female.length)];
  const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
  
  const age = 18 + Math.floor(Math.random() * 72); // 18-90
  const ethnicity = ethnicities[Math.floor(Math.random() * ethnicities.length)];
  const occupation = occupations[Math.floor(Math.random() * occupations.length)];
  
  const pronouns = gender === 'male' ? 'he/him' : 'she/her';
  
  const languageLevels: SPSPatientCase['languageLevel'][] = ['fluent', 'moderate', 'limited'];
  const languageLevel = languageLevels[Math.floor(Math.random() * languageLevels.length)];
  
  const settings: SPSPatientCase['setting'][] = ['clinic', 'emergency_department'];
  const setting = caseTemplate.difficulty === 'advanced' ? 'emergency_department' : settings[Math.floor(Math.random() * settings.length)];
  
  const timeConstraints = [
    '15 minute appointment',
    '30 minute appointment',
    'Busy clinic day',
    'Emergency assessment needed'
  ];
  
  // Generate personality
  const personalities = [
    'anxious and worried',
    'stoic and reserved',
    'talkative and rambling',
    'defensive and skeptical',
    'cooperative and engaged',
    'frustrated with healthcare system',
    'embarrassed about symptoms'
  ];
  
  const emotionalStates = [
    'anxious',
    'calm',
    'distressed',
    'worried',
    'frustrated',
    'scared',
    'embarrassed'
  ];
  
  // Generate medication list (common meds based on age)
  const medications: string[] = [];
  if (age > 50) {
    if (Math.random() > 0.5) medications.push('Atorvastatin 20mg once daily');
    if (Math.random() > 0.5) medications.push('Ramipril 5mg once daily');
    if (Math.random() > 0.7) medications.push('Aspirin 75mg once daily');
  }
  if (gender === 'female' && age < 50 && Math.random() > 0.6) {
    medications.push('Combined oral contraceptive pill');
  }
  if (medications.length === 0) {
    medications.push('None');
  }
  
  // Generate allergies
  const possibleAllergies = ['No known drug allergies', 'Penicillin (rash)', 'Codeine (nausea)', 'NSAIDs (gastric upset)'];
  const allergies = [possibleAllergies[Math.floor(Math.random() * possibleAllergies.length)]];
  
  // Generate PMH
  const commonConditions = ['Hypertension', 'Type 2 diabetes', 'Asthma', 'GERD', 'Osteoarthritis', 'Depression', 'Hypothyroidism'];
  const pastMedicalHistory: string[] = [];
  if (age > 40 && Math.random() > 0.6) {
    pastMedicalHistory.push(commonConditions[Math.floor(Math.random() * commonConditions.length)]);
  }
  if (pastMedicalHistory.length === 0) {
    pastMedicalHistory.push('None significant');
  }
  
  // Generate PSH
  const commonSurgeries = ['Appendectomy', 'Cholecystectomy', 'C-section', 'Hysterectomy', 'Knee arthroscopy'];
  const pastSurgicalHistory: string[] = [];
  if (age > 35 && Math.random() > 0.7) {
    pastSurgicalHistory.push(commonSurgeries[Math.floor(Math.random() * commonSurgeries.length)]);
  }
  if (pastSurgicalHistory.length === 0) {
    pastSurgicalHistory.push('None');
  }
  
  // Social history
  const smokingStatus = ['Non-smoker', '10 cigarettes per day', 'Ex-smoker (quit 5 years ago)', '20 pack-year history'][Math.floor(Math.random() * 4)];
  const alcoholStatus = ['None', 'Social drinker (10 units/week)', 'Occasional', 'Daily (20+ units/week)'][Math.floor(Math.random() * 4)];
  const drugStatus = ['None', 'Recreational cannabis use'][Math.floor(Math.random() * 2)];
  
  const caseId = `SPS-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
  
  return {
    caseId,
    difficulty: caseTemplate.difficulty,
    setting,
    timeConstraint: timeConstraints[Math.floor(Math.random() * timeConstraints.length)],
    
    name: `${firstName} ${lastName}`,
    age,
    gender,
    pronouns,
    ethnicity,
    occupation,
    languageLevel,
    
    chiefComplaint: caseTemplate.chiefComplaint,
    
    onset: caseTemplate.onset,
    location: caseTemplate.location,
    duration: caseTemplate.duration,
    character: caseTemplate.character,
    aggravatingFactors: caseTemplate.aggravatingFactors,
    relievingFactors: caseTemplate.relievingFactors,
    timing: caseTemplate.timing,
    severity: caseTemplate.severity,
    
    vitals: caseTemplate.vitals,
    
    medications,
    allergies,
    pastMedicalHistory,
    pastSurgicalHistory,
    
    familyHistory: age > 40 ? 'Father had heart disease, mother had breast cancer' : 'Non-contributory',
    
    socialHistory: {
      smoking: smokingStatus,
      alcohol: alcoholStatus,
      drugs: drugStatus,
      occupation,
      living: gender === 'male' ? 'Lives with partner' : 'Lives alone'
    },
    
    sexualHistory: gender === 'female' && age < 50 ? 'Sexually active, uses oral contraception' : undefined,
    
    redFlags: caseTemplate.redFlags,
    distractors: caseTemplate.distractors,
    
    physicalExamFindings: caseTemplate.physicalExamFindings,
    
    diagnosis: caseTemplate.diagnosis,
    acceptedDifferentials: caseTemplate.differentials,
    
    personalityTraits: [personalities[Math.floor(Math.random() * personalities.length)]],
    emotionalState: emotionalStates[Math.floor(Math.random() * emotionalStates.length)]
  };
}
