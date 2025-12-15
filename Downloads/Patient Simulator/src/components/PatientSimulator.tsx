import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { generatePatient, PatientProfile } from '../utils/patientGenerator';
import { analyzeConversation, ConversationAnalysis } from '../utils/conversationAnalyzer';
import { getPatientResponse } from '../utils/patientResponses';
import { Send, User, Stethoscope } from 'lucide-react';
import { ScoreCard } from './ScoreCard';

interface Message {
  role: 'patient' | 'provider';
  content: string;
  timestamp: Date;
}

export function PatientSimulator() {
  const [patient] = useState<PatientProfile>(() => generatePatient());
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentInput, setCurrentInput] = useState('');
  const [conversationComplete, setConversationComplete] = useState(false);
  const [analysis, setAnalysis] = useState<ConversationAnalysis | null>(null);
  const [turnCount, setTurnCount] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Patient's opening statement
    const openingMessage: Message = {
      role: 'patient',
      content: patient.openingStatement,
      timestamp: new Date()
    };
    setMessages([openingMessage]);
  }, [patient]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = () => {
    if (!currentInput.trim() || conversationComplete) return;

    const newMessage: Message = {
      role: 'provider',
      content: currentInput,
      timestamp: new Date()
    };

    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setCurrentInput('');

    const newTurnCount = turnCount + 1;
    setTurnCount(newTurnCount);

    // Get patient response
    const patientResponse = getPatientResponse(
      patient,
      currentInput,
      updatedMessages,
      newTurnCount
    );

    setTimeout(() => {
      const responseMessage: Message = {
        role: 'patient',
        content: patientResponse.content,
        timestamp: new Date()
      };

      const finalMessages = [...updatedMessages, responseMessage];
      setMessages(finalMessages);

      // Check if conversation should end
      if (patientResponse.shouldEnd || newTurnCount >= 8) {
        setConversationComplete(true);
        
        // Analyze conversation
        const conversationAnalysis = analyzeConversation(
          finalMessages,
          patient
        );
        setAnalysis(conversationAnalysis);
      }
    }, 800);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleEndConversation = () => {
    if (messages.length > 0) {
      setConversationComplete(true);
      const conversationAnalysis = analyzeConversation(
        messages,
        patient
      );
      setAnalysis(conversationAnalysis);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Patient Info Panel */}
      <div className="lg:col-span-1">
        <Card>
          <CardHeader>
            <CardTitle>Patient Information</CardTitle>
            <CardDescription>Current scenario details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-sm text-gray-600 mb-1">Name</p>
              <p className="text-gray-900">{patient.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-1">Age</p>
              <p className="text-gray-900">{patient.age} years old</p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-1">Chief Concern</p>
              <p className="text-gray-900">{patient.concern}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-1">Emotional State</p>
              <Badge variant="outline" className="border-primary text-primary">{patient.emotionalState}</Badge>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-1">Communication Challenge</p>
              <p className="text-sm text-gray-900">{patient.communicationChallenge}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-1">Background</p>
              <p className="text-sm text-gray-900">{patient.background}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chat Interface */}
      <div className="lg:col-span-2">
        <Card className="h-[700px] flex flex-col">
          <CardHeader>
            <CardTitle>Patient Conversation</CardTitle>
            <CardDescription>
              {conversationComplete 
                ? 'Conversation complete - Review your score below'
                : 'Respond to the patient using effective communication techniques'
              }
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col p-0">
            {/* Messages */}
            <ScrollArea className="flex-1 px-6" ref={scrollRef}>
              <div className="space-y-4 pb-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex gap-3 ${
                      message.role === 'provider' ? 'flex-row-reverse' : ''
                    }`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.role === 'patient' 
                        ? 'bg-primary/10 text-primary' 
                        : 'bg-accent/10 text-accent'
                    }`}>
                      {message.role === 'patient' ? (
                        <User className="w-4 h-4" />
                      ) : (
                        <Stethoscope className="w-4 h-4" />
                      )}
                    </div>
                    <div
                      className={`rounded-lg px-4 py-3 max-w-[80%] ${
                        message.role === 'patient'
                          ? 'bg-gray-50 border border-gray-200'
                          : 'bg-teal-50 border border-teal-200'
                      }`}
                    >
                      <p className="text-sm text-gray-900">{message.content}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>

            {/* Score Card */}
            {conversationComplete && analysis && (
              <div className="px-6 pb-4">
                <ScoreCard analysis={analysis} patient={patient} />
              </div>
            )}

            {/* Input Area */}
            {!conversationComplete && (
              <div className="border-t p-4">
                <div className="flex gap-2">
                  <Textarea
                    value={currentInput}
                    onChange={(e) => setCurrentInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your response to the patient..."
                    className="min-h-[80px]"
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={!currentInput.trim()}
                    className="self-end bg-accent hover:bg-accent/90 text-white"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
                <div className="flex items-center justify-between mt-3">
                  <p className="text-xs text-gray-600">
                    Press Enter to send • Shift+Enter for new line
                  </p>
                  {messages.length > 1 && (
                    <Button
                      onClick={handleEndConversation}
                      variant="outline"
                      size="sm"
                      className="border-primary text-primary hover:bg-primary/10"
                    >
                      End & Get Grade
                    </Button>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
