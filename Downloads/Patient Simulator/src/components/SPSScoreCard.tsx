import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Button } from './ui/button';
import { CheckCircle2, AlertCircle, XCircle, ChevronDown, ChevronUp, Award } from 'lucide-react';
import { SPSScore } from '../utils/spsScoring';
import { useState } from 'react';

interface SPSScoreCardProps {
  score: SPSScore;
}

export function SPSScoreCard({ score }: SPSScoreCardProps) {
  const [expandedRubrics, setExpandedRubrics] = useState<Set<string>>(new Set());

  const toggleRubric = (rubric: string) => {
    setExpandedRubrics(prev => {
      const next = new Set(prev);
      if (next.has(rubric)) {
        next.delete(rubric);
      } else {
        next.add(rubric);
      }
      return next;
    });
  };

  const getScoreColor = (percentage: number) => {
    if (percentage >= 90) return 'text-green-600';
    if (percentage >= 75) return 'text-accent';
    if (percentage >= 60) return 'text-orange-500';
    return 'text-primary';
  };

  const getScoreBadgeVariant = (percentage: number): "default" | "secondary" | "outline" => {
    if (percentage >= 80) return 'default';
    if (percentage >= 60) return 'secondary';
    return 'outline';
  };

  const rubricEntries = [
    { key: 'historyGathering', label: 'History Gathering', data: score.rubrics.historyGathering },
    { key: 'clinicalReasoning', label: 'Clinical Reasoning', data: score.rubrics.clinicalReasoning },
    { key: 'diagnosisAssessment', label: 'Diagnosis/Assessment', data: score.rubrics.diagnosisAssessment },
    { key: 'communicationEmpathy', label: 'Communication & Empathy', data: score.rubrics.communicationEmpathy },
    { key: 'safetyRedFlags', label: 'Safety & Red Flags', data: score.rubrics.safetyRedFlags },
    { key: 'counselingNextSteps', label: 'Counseling & Next Steps', data: score.rubrics.counselingNextSteps },
    { key: 'timeManagement', label: 'Time Management', data: score.rubrics.timeManagement }
  ];

  return (
    <Card className="border-2 border-accent/20 bg-gradient-to-br from-white to-teal-50/30">
      <CardHeader className="border-b border-gray-200 bg-white">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 mb-2">
              <Award className="w-5 h-5 text-accent" />
              Clinical Encounter Scorecard
            </CardTitle>
            <p className="text-sm text-gray-600">Case ID: {score.caseId}</p>
            <p className="text-sm text-gray-700 mt-1">{score.patientSummary}</p>
          </div>
          <div className="text-right">
            <div className={`${getScoreColor(score.totalScore)}`}>
              <p className="text-sm text-gray-600">Overall Score</p>
              <p className="text-4xl">{score.totalScore}</p>
              <p className="text-sm">/ 100</p>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6 pt-6">
        {/* Rubric Breakdown */}
        <div>
          <h3 className="text-sm mb-3 text-gray-900">Performance by Rubric</h3>
          <div className="space-y-2">
            {rubricEntries.map(({ key, label, data }) => {
              const percentage = (data.score / data.maxScore) * 100;
              const isExpanded = expandedRubrics.has(key);

              return (
                <Collapsible key={key} open={isExpanded} onOpenChange={() => toggleRubric(key)}>
                  <div className="border border-gray-200 rounded-lg p-3 bg-white">
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" className="w-full p-0 h-auto hover:bg-transparent">
                        <div className="w-full">
                          <div className="flex justify-between items-center mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-gray-900">{label}</span>
                              {isExpanded ? (
                                <ChevronUp className="w-4 h-4 text-gray-600" />
                              ) : (
                                <ChevronDown className="w-4 h-4 text-gray-600" />
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-gray-700">
                                {data.score}/{data.maxScore}
                              </span>
                              <Badge variant={getScoreBadgeVariant(percentage)} className="text-xs">
                                {data.weight.toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                          <Progress value={percentage} className="h-2" />
                        </div>
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-3 space-y-2">
                      <div className="text-xs bg-slate-50 p-3 rounded border border-gray-200">
                        <p className="text-gray-900 mb-2">{data.feedback}</p>
                        {data.evidence.length > 0 && (
                          <div className="mt-2">
                            <p className="text-gray-700 mb-1">Evidence:</p>
                            <ul className="list-disc list-inside text-gray-600 space-y-1">
                              {data.evidence.map((ev, idx) => (
                                <li key={idx}>{ev}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              );
            })}
          </div>
        </div>

        {/* Ground Truth Diagnosis */}
        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-sm mb-2 text-gray-900">Ground Truth Diagnosis</h3>
          <div className="bg-teal-50 border border-teal-200 rounded p-3">
            <p className="text-teal-900">{score.groundTruthDiagnosis}</p>
            <p className="text-xs text-teal-700 mt-2">Accepted differentials: {score.acceptedDifferentials.join(', ')}</p>
          </div>
        </div>

        {/* Quoted Evidence */}
        {score.quotedEvidence.length > 0 && (
          <div>
            <h3 className="text-sm mb-2 text-gray-900">Evidence from Interview</h3>
            <div className="space-y-2">
              {score.quotedEvidence.slice(0, 5).map((evidence, idx) => (
                <div key={idx} className="bg-slate-50 border border-gray-200 rounded p-3 text-xs">
                  <p className="text-gray-900 italic mb-1">"{evidence.quote}"</p>
                  <p className="text-gray-600">
                    <span className="text-accent">→</span> {evidence.explanation}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Strengths */}
        <div>
          <h3 className="text-sm mb-2 flex items-center gap-2 text-gray-900">
            <CheckCircle2 className="w-4 h-4 text-green-600" />
            Strengths
          </h3>
          <ul className="space-y-1">
            {score.strengths.map((strength, index) => (
              <li key={index} className="text-sm text-gray-700 pl-6">
                • {strength}
              </li>
            ))}
          </ul>
        </div>

        {/* Areas for Improvement */}
        <div>
          <h3 className="text-sm mb-2 flex items-center gap-2 text-gray-900">
            <AlertCircle className="w-4 h-4 text-accent" />
            Areas for Improvement
          </h3>
          <ul className="space-y-1">
            {score.improvements.map((improvement, index) => (
              <li key={index} className="text-sm text-gray-700 pl-6">
                • {improvement}
              </li>
            ))}
          </ul>
        </div>

        {/* Missed Opportunities */}
        {score.missedOpportunities.length > 0 && (
          <div>
            <h3 className="text-sm mb-2 flex items-center gap-2 text-gray-900">
              <XCircle className="w-4 h-4 text-primary" />
              Missed Opportunities
            </h3>
            <ul className="space-y-1">
              {score.missedOpportunities.map((missed, index) => (
                <li key={index} className="text-sm text-gray-700 pl-6">
                  • {missed}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Feedback Note */}
        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-sm mb-2 text-gray-900">Feedback Note</h3>
          <div className="bg-slate-50 border border-gray-200 rounded p-4">
            <p className="text-sm text-gray-700 whitespace-pre-wrap">{score.feedbackNote}</p>
          </div>
        </div>

        {/* Suggested Focus */}
        <div className="bg-accent/5 border border-accent/20 rounded p-4">
          <h3 className="text-sm mb-2 text-gray-900">Suggested Practice Focus</h3>
          <p className="text-sm text-gray-700">{score.suggestedFocus}</p>
        </div>
      </CardContent>
    </Card>
  );
}
