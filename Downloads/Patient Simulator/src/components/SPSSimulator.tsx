import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Send, User, Stethoscope, AlertCircle } from 'lucide-react';
import { generateSPSPatient, SPSPatientCase } from '../utils/spsPatientGenerator';
import { getSPSPatientResponse } from '../utils/spsPatientResponses';
import { SPSScoreCard } from './SPSScoreCard';
import { scoreSPSCase, SPSScore } from '../utils/spsScoring';

interface Message {
  role: 'patient' | 'provider';
  content: string;
  timestamp: Date;
}

export function SPSSimulator() {
  const [patient, setPatient] = useState<SPSPatientCase | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentInput, setCurrentInput] = useState('');
  const [isPatientTyping, setIsPatientTyping] = useState(false);
  const [conversationEnded, setConversationEnded] = useState(false);
  const [score, setScore] = useState<SPSScore | null>(null);
  const [isScoring, setIsScoring] = useState(false);
  const [startTime] = useState<Date>(new Date());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Generate initial patient case
    const newPatient = generateSPSPatient();
    setPatient(newPatient);
    
    // Initial patient greeting
    const greeting = getInitialGreeting(newPatient);
    setMessages([{
      role: 'patient',
      content: greeting,
      timestamp: new Date()
    }]);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const getInitialGreeting = (pt: SPSPatientCase): string => {
    if (pt.setting === 'emergency_department') {
      return `*${pt.name} looks ${pt.emotionalState}* Hi doctor, I'm here because of ${pt.chiefComplaint}.`;
    }
    return `*${pt.name} appears ${pt.emotionalState}* Hello, thanks for seeing me. I've been having ${pt.chiefComplaint}.`;
  };

  const handleSendMessage = async () => {
    if (!currentInput.trim() || !patient || conversationEnded) return;

    const providerMessage: Message = {
      role: 'provider',
      content: currentInput,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, providerMessage]);
    setCurrentInput('');
    setIsPatientTyping(true);

    // Check if user is asking to end/grade
    if (/\b(grade|end case|finish|score me)\b/i.test(currentInput)) {
      await handleEndAndGrade();
      return;
    }

    try {
      const response = await getSPSPatientResponse(
        patient,
        currentInput,
        [...messages, providerMessage]
      );

      setTimeout(() => {
        const patientMessage: Message = {
          role: 'patient',
          content: response.content,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, patientMessage]);
        setIsPatientTyping(false);

        if (response.shouldEnd) {
          handleEndAndGrade();
        }
      }, 800 + Math.random() * 1200);
    } catch (error) {
      console.error('Error getting patient response:', error);
      setIsPatientTyping(false);
    }
  };

  const handleEndAndGrade = async () => {
    if (!patient || conversationEnded) return;
    
    setConversationEnded(true);
    setIsScoring(true);
    
    try {
      const finalScore = await scoreSPSCase(patient, messages);
      setScore(finalScore);
    } catch (error) {
      console.error('Error scoring case:', error);
    } finally {
      setIsScoring(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!patient) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-600">Generating patient case...</p>
        </div>
      </div>
    );
  }

  const getElapsedTime = () => {
    const now = new Date();
    const diffMs = now.getTime() - startTime.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    return diffMins;
  };

  return (
    <div className="space-y-4">
      {/* Patient Info Card */}
      <Card className="border-2 border-gray-200">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between">
            <span>Patient Chart</span>
            <div className="flex gap-2">
              <Badge variant="outline" className="text-xs">
                {patient.setting === 'emergency_department' ? 'Emergency Department' : 'Outpatient Clinic'}
              </Badge>
              <Badge variant="outline" className="text-xs">
                {patient.difficulty}
              </Badge>
              <Badge variant="outline" className="text-xs">
                {getElapsedTime()} min
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-600 mb-1">Name</p>
              <p className="text-gray-900">{patient.name}</p>
            </div>
            <div>
              <p className="text-gray-600 mb-1">Age / Gender</p>
              <p className="text-gray-900">{patient.age} / {patient.gender}</p>
            </div>
            <div>
              <p className="text-gray-600 mb-1">Chief Complaint</p>
              <p className="text-gray-900">{patient.chiefComplaint}</p>
            </div>
            {!conversationEnded && (
              <>
                <div className="col-span-2 md:col-span-3">
                  <div className="bg-amber-50 border border-amber-200 rounded p-3 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                    <div className="text-xs text-amber-900">
                      <p className="mb-1">This is a clinical simulation. Conduct a focused history and examination as you would in real practice.</p>
                      <p>Type "grade" or "end case" when you're ready for evaluation.</p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Conversation Area */}
      {!score ? (
        <Card>
          <CardHeader>
            <CardTitle>Clinical Encounter</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Messages */}
            <div className="h-[400px] overflow-y-auto space-y-4 p-4 bg-slate-50 rounded-lg">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-3 ${message.role === 'provider' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.role === 'patient' && (
                    <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-primary/10 text-primary">
                      <User className="w-4 h-4" />
                    </div>
                  )}
                  <div
                    className={`rounded-lg px-4 py-3 max-w-[80%] ${
                      message.role === 'patient'
                        ? 'bg-white border border-gray-200'
                        : 'bg-accent text-white'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  </div>
                  {message.role === 'provider' && (
                    <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-accent/10 text-accent">
                      <Stethoscope className="w-4 h-4" />
                    </div>
                  )}
                </div>
              ))}
              {isPatientTyping && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center bg-primary/10 text-primary">
                    <User className="w-4 h-4" />
                  </div>
                  <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            {!conversationEnded && (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <Textarea
                    value={currentInput}
                    onChange={(e) => setCurrentInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Ask questions, perform examination, or provide counseling..."
                    className="flex-1 min-h-[60px]"
                    disabled={isPatientTyping}
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={!currentInput.trim() || isPatientTyping}
                    className="self-end bg-accent hover:bg-accent/90 text-white"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
                <div className="flex items-center justify-between">
                  <p className="text-xs text-gray-600">
                    Press Enter to send • Shift+Enter for new line
                  </p>
                  {messages.length > 2 && (
                    <Button
                      onClick={handleEndAndGrade}
                      variant="outline"
                      size="sm"
                      className="border-primary text-primary hover:bg-primary/10"
                    >
                      End Case & Get Grade
                    </Button>
                  )}
                </div>
              </div>
            )}

            {isScoring && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                <p className="text-gray-700">Evaluating your clinical encounter...</p>
                <p className="text-sm text-gray-500 mt-2">Analyzing history gathering, clinical reasoning, and communication skills</p>
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <SPSScoreCard score={score} />
      )}
    </div>
  );
}
