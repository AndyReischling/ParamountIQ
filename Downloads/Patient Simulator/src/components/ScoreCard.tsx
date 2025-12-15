import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { ConversationAnalysis } from '../utils/conversationAnalyzer';
import { PatientProfile } from '../utils/patientGenerator';
import { CheckCircle2, XCircle, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Button } from './ui/button';

interface ScoreCardProps {
  analysis: ConversationAnalysis;
  patient: PatientProfile;
}

export function ScoreCard({ analysis }: ScoreCardProps) {
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    start: false,
    heart: false,
    care: false,
    wow: false
  });

  const toggleSection = (section: string) => {
    setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 75) return 'text-accent';
    if (score >= 60) return 'text-orange-500';
    return 'text-primary';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 90) return 'Excellent';
    if (score >= 75) return 'Good';
    if (score >= 60) return 'Needs Improvement';
    return 'Poor';
  };

  return (
    <Card className="border-2 border-teal-200 bg-teal-50/30">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Communication Skills Assessment</CardTitle>
            <CardDescription>Based on S.T.A.R.T., H.E.A.R.T., C.A.R.E., and WOW frameworks</CardDescription>
          </div>
          <div className="text-center">
            <div className={`text-4xl ${getScoreColor(analysis.totalScore)}`}>
              {analysis.totalScore}
            </div>
            <Badge variant="outline" className="mt-1">
              {getScoreLabel(analysis.totalScore)}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Category Scores with Detailed Breakdown */}
        <div className="space-y-3">
          {/* S.T.A.R.T. */}
          <Collapsible open={openSections.start} onOpenChange={() => toggleSection('start')}>
            <div className="border border-gray-200 rounded-lg p-3 bg-white">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full p-0 h-auto hover:bg-transparent">
                  <div className="w-full">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-900">S.T.A.R.T. Framework</span>
                        {openSections.start ? <ChevronUp className="w-4 h-4 text-gray-600" /> : <ChevronDown className="w-4 h-4 text-gray-600" />}
                      </div>
                      <span className="text-sm text-gray-700">{analysis.scores.start}/25</span>
                    </div>
                    <Progress value={(analysis.scores.start / 25) * 100} />
                  </div>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 space-y-2">
                {analysis.scoringDetails.start.map((detail, idx) => (
                  <div key={idx} className="text-xs bg-slate-50 p-2 rounded border border-gray-200">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-gray-900">{detail.element}</span>
                      <Badge variant={detail.found ? "default" : "outline"} className="text-xs">
                        {detail.earnedPoints}/{detail.maxPoints} pts
                      </Badge>
                    </div>
                    <p className="text-gray-700">{detail.explanation}</p>
                  </div>
                ))}
              </CollapsibleContent>
            </div>
          </Collapsible>

          {/* H.E.A.R.T. */}
          <Collapsible open={openSections.heart} onOpenChange={() => toggleSection('heart')}>
            <div className="border border-gray-200 rounded-lg p-3 bg-white">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full p-0 h-auto hover:bg-transparent">
                  <div className="w-full">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-900">H.E.A.R.T. Framework</span>
                        {openSections.heart ? <ChevronUp className="w-4 h-4 text-gray-600" /> : <ChevronDown className="w-4 h-4 text-gray-600" />}
                      </div>
                      <span className="text-sm text-gray-700">{analysis.scores.heart}/25</span>
                    </div>
                    <Progress value={(analysis.scores.heart / 25) * 100} />
                  </div>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 space-y-2">
                {analysis.scoringDetails.heart.map((detail, idx) => (
                  <div key={idx} className="text-xs bg-slate-50 p-2 rounded border border-gray-200">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-gray-900">{detail.element}</span>
                      <Badge variant={detail.found ? "default" : "outline"} className="text-xs">
                        {detail.earnedPoints}/{detail.maxPoints} pts
                      </Badge>
                    </div>
                    <p className="text-gray-700">{detail.explanation}</p>
                  </div>
                ))}
              </CollapsibleContent>
            </div>
          </Collapsible>

          {/* C.A.R.E. */}
          <Collapsible open={openSections.care} onOpenChange={() => toggleSection('care')}>
            <div className="border border-gray-200 rounded-lg p-3 bg-white">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full p-0 h-auto hover:bg-transparent">
                  <div className="w-full">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-900">C.A.R.E. Behaviors</span>
                        {openSections.care ? <ChevronUp className="w-4 h-4 text-gray-600" /> : <ChevronDown className="w-4 h-4 text-gray-600" />}
                      </div>
                      <span className="text-sm text-gray-700">{analysis.scores.care}/25</span>
                    </div>
                    <Progress value={(analysis.scores.care / 25) * 100} />
                  </div>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 space-y-2">
                {analysis.scoringDetails.care.map((detail, idx) => (
                  <div key={idx} className="text-xs bg-slate-50 p-2 rounded border border-gray-200">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-gray-900">{detail.element}</span>
                      <Badge variant={detail.found ? "default" : "outline"} className="text-xs">
                        {detail.earnedPoints}/{detail.maxPoints} pts
                      </Badge>
                    </div>
                    <p className="text-gray-700">{detail.explanation}</p>
                  </div>
                ))}
              </CollapsibleContent>
            </div>
          </Collapsible>

          {/* WOW */}
          <Collapsible open={openSections.wow} onOpenChange={() => toggleSection('wow')}>
            <div className="border border-gray-200 rounded-lg p-3 bg-white">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full p-0 h-auto hover:bg-transparent">
                  <div className="w-full">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-900">WOW Communication</span>
                        {openSections.wow ? <ChevronUp className="w-4 h-4 text-gray-600" /> : <ChevronDown className="w-4 h-4 text-gray-600" />}
                      </div>
                      <span className="text-sm text-gray-700">{analysis.scores.wow}/25</span>
                    </div>
                    <Progress value={(analysis.scores.wow / 25) * 100} />
                  </div>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 space-y-2">
                {analysis.scoringDetails.wow.map((detail, idx) => (
                  <div key={idx} className="text-xs bg-slate-50 p-2 rounded border border-gray-200">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-gray-900">{detail.element}</span>
                      <Badge variant={detail.found ? "default" : "outline"} className="text-xs">
                        {detail.earnedPoints}/{detail.maxPoints} pts
                      </Badge>
                    </div>
                    <p className="text-gray-700">{detail.explanation}</p>
                  </div>
                ))}
              </CollapsibleContent>
            </div>
          </Collapsible>
        </div>

        {/* Strengths */}
        <div>
          <h4 className="text-sm mb-2 flex items-center gap-2 text-gray-900">
            <CheckCircle2 className="w-4 h-4 text-green-600" />
            Strengths
          </h4>
          <ul className="space-y-1">
            {analysis.feedback.strengths.map((strength, index) => (
              <li key={index} className="text-sm text-gray-700 pl-6">
                • {strength}
              </li>
            ))}
          </ul>
        </div>

        {/* Areas for Improvement */}
        <div>
          <h4 className="text-sm mb-2 flex items-center gap-2 text-gray-900">
            <AlertCircle className="w-4 h-4 text-accent" />
            Areas for Improvement
          </h4>
          <ul className="space-y-1">
            {analysis.feedback.improvements.map((improvement, index) => (
              <li key={index} className="text-sm text-gray-700 pl-6">
                • {improvement}
              </li>
            ))}
          </ul>
        </div>

        {/* Missed Opportunities */}
        {analysis.feedback.missed.length > 0 && (
          <div>
            <h4 className="text-sm mb-2 flex items-center gap-2 text-gray-900">
              <XCircle className="w-4 h-4 text-primary" />
              Missed Opportunities
            </h4>
            <ul className="space-y-1">
              {analysis.feedback.missed.map((missed, index) => (
                <li key={index} className="text-sm text-gray-700 pl-6">
                  • {missed}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Overall Assessment */}
        <div className="pt-4 border-t border-gray-200">
          <h4 className="text-sm mb-2 text-gray-900">Overall Assessment</h4>
          <p className="text-sm text-gray-700">{analysis.feedback.overall}</p>
        </div>
      </CardContent>
    </Card>
  );
}
