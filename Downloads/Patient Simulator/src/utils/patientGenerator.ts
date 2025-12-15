export interface PatientProfile {
  id: string;
  name: string;
  age: number;
  gender: string;
  concern: string;
  emotionalState: string;
  communicationChallenge: string;
  background: string;
  openingStatement: string;
  resolution: string;
  personalityTraits: string[];
}

const firstNames = {
  male: ['James', 'Robert', 'Michael', 'David', 'John', 'William', 'Richard', 'Thomas', 'Carlos', 'Jose'],
  female: ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Maria', 'Susan', 'Jessica', 'Sarah', 'Karen']
};

const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Wilson', 'Anderson', 'Taylor', 'Thomas', 'Moore', 'Jackson', 'Lee', 'Patel'];

const concerns = [
  { concern: 'chronic lower back pain radiating down left leg', severity: 'moderate', medicalContext: 'worsening over 6 months, difficulty standing for more than 20 minutes' },
  { concern: 'newly diagnosed Type 2 diabetes with HbA1c of 8.2', severity: 'high', medicalContext: 'struggling to adjust diet and confused about blood glucose monitoring' },
  { concern: 'severe nausea and dizziness from new hypertension medication', severity: 'moderate', medicalContext: 'started lisinopril 2 weeks ago, symptoms interfering with work' },
  { concern: 'abnormal mammogram results showing suspicious calcifications', severity: 'high', medicalContext: 'recommended for biopsy, extremely worried about breast cancer' },
  { concern: 'persistent wound infection after gallbladder surgery', severity: 'high', medicalContext: 'incision site red and draining, surgery was 3 weeks ago' },
  { concern: 'insurance denied coverage for recommended MRI', severity: 'moderate', medicalContext: 'cannot afford $3,200 out of pocket, symptoms continue' },
  { concern: 'waited 4 months for endocrinology appointment', severity: 'low', medicalContext: 'thyroid nodules found, frustrated by healthcare system delays' },
  { concern: 'conflicting treatment plans from PCP and cardiologist', severity: 'moderate', medicalContext: 'one wants statin, other says wait, feeling caught in the middle' },
  { concern: 'father diagnosed with stage 4 lung cancer', severity: 'high', medicalContext: 'need guidance on hospice vs continued treatment, family disagreement' },
  { concern: 'family history of colon cancer, turning 45 next month', severity: 'low', medicalContext: 'wants to discuss colonoscopy screening and prevention' },
  { concern: 'daily migraines with aura for past 3 months', severity: 'moderate', medicalContext: 'missing 2-3 days of work per week, tried OTC meds without relief' },
  { concern: 'severe anxiety about upcoming knee replacement surgery', severity: 'high', medicalContext: 'scheduled in 2 weeks, never had surgery before, catastrophizing complications' },
  { concern: 'gained 40 pounds since starting antidepressants', severity: 'moderate', medicalContext: 'tried multiple diets, weight affecting self-esteem and mobility' },
  { concern: 'chronic insomnia, sleeping only 3-4 hours per night', severity: 'moderate', medicalContext: 'exhausted, affecting job performance and relationships' },
  { concern: 'chest pain and palpitations, ER said "anxiety"', severity: 'moderate', medicalContext: 'symptoms feel real, frustrated by dismissal, wants cardiac workup' },
  { concern: 'recurrent UTIs, 5 episodes in past 6 months', severity: 'moderate', medicalContext: 'on third round of antibiotics, concerned about antibiotic resistance' },
  { concern: 'shortness of breath worsening over 2 months', severity: 'high', medicalContext: 'difficulty climbing stairs, 30-year smoking history, worried about COPD' },
  { concern: 'elevated PSA level at 6.8, recommended for biopsy', severity: 'high', medicalContext: 'terrified of prostate cancer diagnosis and potential impotence' }
];

const emotionalStates = [
  'Anxious',
  'Frustrated',
  'Angry',
  'Confused',
  'Worried',
  'Scared',
  'Defeated',
  'Hopeful',
  'Skeptical',
  'Overwhelmed',
  'Defensive',
  'Impatient',
  'Tearful',
  'Resigned'
];

const communicationChallenges = [
  'Tends to interrupt and talk over others',
  'Very quiet and reluctant to share information',
  'Brings up multiple unrelated concerns',
  'Questions every recommendation',
  'Uses medical jargon incorrectly',
  'Has limited health literacy',
  'Cultural/language barriers present',
  'Has had negative past healthcare experiences',
  'Accompanied by overbearing family member',
  'Distrustful of medical establishment',
  'Rambles and struggles to stay on topic',
  'Minimizes symptoms or concerns',
  'Highly emotional and tearful',
  'Defensive about lifestyle choices',
  'Demands specific treatments or tests'
];

const personalityOptions = [
  ['direct', 'task-oriented', 'independent'],
  ['expressive', 'emotional', 'talkative'],
  ['analytical', 'detail-oriented', 'cautious'],
  ['amiable', 'cooperative', 'relationship-focused'],
  ['skeptical', 'questioning', 'research-oriented'],
  ['passive', 'agreeable', 'conflict-averse'],
  ['assertive', 'confident', 'demanding'],
  ['anxious', 'worried', 'catastrophizing']
];

export function generatePatient(): PatientProfile {
  const gender = Math.random() > 0.5 ? 'male' : 'female';
  const firstName = firstNames[gender][Math.floor(Math.random() * firstNames[gender].length)];
  const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
  const age = Math.floor(Math.random() * 60) + 20; // 20-80 years old
  
  const concernData = concerns[Math.floor(Math.random() * concerns.length)];
  const emotionalState = emotionalStates[Math.floor(Math.random() * emotionalStates.length)];
  const communicationChallenge = communicationChallenges[Math.floor(Math.random() * communicationChallenges.length)];
  const personalityTraits = personalityOptions[Math.floor(Math.random() * personalityOptions.length)];

  // Generate background
  const backgrounds = [
    `High school math teacher, married with two teenagers, limited time off during school year, relies on spouse's insurance`,
    `Retired postal worker age 68, widowed 2 years ago, lives alone in apartment, adult children live in other states, on Medicare`,
    `Owns a local bakery, works 60+ hours/week, no employees so can't take time off, uninsured until recently`,
    `Full-time caregiver for mother with Alzheimer's, extremely stressed, neglecting own health needs, Medicaid for mother`,
    `College sophomore living on campus, first time navigating healthcare without parents, on parents' insurance until age 26`,
    `Single mother of 7-year-old twins, works two part-time jobs, struggles to afford copays, recently qualified for Medicaid`,
    `Emergency room nurse for 15 years, works nights, delays seeking care due to knowing "too much", has good insurance through hospital`,
    `Software engineer, relocated from California 6 months ago, trying to establish care with new providers, excellent insurance`,
    `Army veteran age 52, served in Iraq, receives VA benefits, prefers direct communication, distrustful of some medical advice`,
    `Overnight warehouse supervisor, sleeps during day, difficult to schedule appointments, wife nags about health issues`,
    `Freelance graphic designer, works from home, anxiety about medical settings, individual marketplace insurance with high deductible`,
    `Restaurant server, inconsistent schedule, recently lost insurance when hours were cut, worried about costs`,
    `Accountant in tax season, extremely busy January-April, putting off care until work slows down, has HSA account`,
    `Stay-at-home parent of infant and preschooler, feels guilty taking time for own health, on spouse's insurance plan`,
    `Construction worker, physical job, fears diagnosis might mean disability/job loss, workers' comp claim denied previously`,
    `Graduate student in social work, very limited income, university student health plan with restrictions on specialists`
  ];
  const background = backgrounds[Math.floor(Math.random() * backgrounds.length)];

  // Generate opening statement based on emotional state and concern
  const openingStatements = generateOpeningStatement(firstName, concernData, emotionalState, communicationChallenge);
  
  return {
    id: `patient_${Date.now()}_${Math.random().toString(36).substring(7)}`,
    name: `${firstName} ${lastName}`,
    age,
    gender,
    concern: concernData.concern,
    emotionalState,
    communicationChallenge,
    background,
    openingStatement: openingStatements,
    resolution: generateResolution(concernData),
    personalityTraits
  };
}

function generateOpeningStatement(name: string, concernData: any, emotionalState: string, challenge: string): string {
  const statements = [
    `Hi... I've been dealing with ${concernData.concern} and I'm really ${emotionalState.toLowerCase()}. I don't even know where to start.`,
    `Look, I need to talk to someone about ${concernData.concern}. This has been going on for ${concernData.medicalContext} and I'm ${emotionalState.toLowerCase()}.`,
    `I'm here because... well, I have ${concernData.concern}. I'm feeling pretty ${emotionalState.toLowerCase()} about the whole situation.`,
    `Excuse me, I've been waiting to discuss ${concernData.concern}. I'm ${emotionalState.toLowerCase()} and I need some answers.`,
    `Hi, sorry, I'm a bit ${emotionalState.toLowerCase()}. I'm dealing with ${concernData.concern} and I'm not sure what to do.`
  ];
  
  return statements[Math.floor(Math.random() * statements.length)];
}

function generateResolution(concernData: any): string {
  const resolutions = [
    `Understanding the treatment plan and next steps`,
    `Feeling heard and having concerns validated`,
    `Getting a referral to a specialist`,
    `Receiving clear information about diagnosis`,
    `Having medication adjusted or changed`,
    `Scheduling follow-up care`,
    `Getting resources for financial assistance`,
    `Understanding what to expect from upcoming procedure`
  ];
  
  return resolutions[Math.floor(Math.random() * resolutions.length)];
}
