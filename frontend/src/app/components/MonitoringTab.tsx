import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { BarChart3, Activity, TrendingUp, AlertCircle, RefreshCw, Server, Cpu } from "lucide-react";
import { apiFetch } from "../../lib/auth";

interface MetricSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

interface MetricFamily {
  name: string;
  type: string;
  samples: MetricSample[];
}

export function MonitoringTab() {
  const [metrics, setMetrics] = useState<Record<string, MetricFamily> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadMetrics = async () => {
    try {
      const r = await apiFetch("/api/admin/monitoring");
      if (!r.ok) throw new Error("Failed to load metrics");
      const data = await r.json();
      setMetrics(data);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
    if (!autoRefresh) return;
    const interval = setInterval(loadMetrics, 15000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const findMetric = (name: string): MetricSample[] => {
    if (!metrics || !metrics[name]) return [];
    return metrics[name].samples;
  };

  const sumMetric = (name: string): number => {
    return findMetric(name).reduce((s, m) => s + m.value, 0);
  };

  const avgMetric = (name: string): number => {
    const samples = findMetric(name);
    if (samples.length === 0) return 0;
    return samples.reduce((s, m) => s + m.value, 0) / samples.length;
  };

  const pipelineErrors = findMetric("pipeline_errors_total");
  const ltrMetrics = findMetric("ltr_model_metric");
  const pipelineDurationSamples = findMetric("pipeline_stage_duration_seconds_count");
  const recsGenerated = findMetric("recommendations_generated_total");
  const apiReqCount = findMetric("api_requests_total");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Мониторинг</h2>
          <p className="text-sm text-gray-500">Метрики системы в реальном времени</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} className="rounded" />
            Автообновление
          </label>
          <Button variant="outline" size="sm" onClick={loadMetrics} disabled={loading}>
            <RefreshCw className={`size-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Обновить
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="size-5" />
          <span>{error}</span>
        </div>
      )}

      {loading && !metrics && (
        <div className="text-center py-12 text-gray-500">Загрузка метрик...</div>
      )}

      {!loading && metrics && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2">
                  <Activity className="size-4" />
                  Pipeline runs
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{sumMetric("pipeline_stage_duration_seconds_count")}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2">
                  <AlertCircle className="size-4" />
                  Pipeline errors
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600">{pipelineErrors.length > 0 ? pipelineErrors.reduce((s, m) => s + m.value, 0) : 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2">
                  <TrendingUp className="size-4" />
                  Recommendations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">{recsGenerated.length > 0 ? recsGenerated.reduce((s, m) => s + m.value, 0).toFixed(0) : 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2">
                  <Server className="size-4" />
                  API requests
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-600">{apiReqCount.length > 0 ? apiReqCount.reduce((s, m) => s + m.value, 0).toFixed(0) : 0}</div>
              </CardContent>
            </Card>
          </div>

          {/* Pipeline stage durations */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <BarChart3 className="size-5" />
                Pipeline stages
              </CardTitle>
              <CardDescription>Длительность этапов пайплайна</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(["data_collection", "quality_scoring", "skill_extraction", "weight_cleaning", "level_building", "cluster_training", "model_training", "gap_analysis"] as const).map((stage) => {
                  const countSamples = findMetric("pipeline_stage_duration_seconds_count").filter(s => s.labels.stage === stage);
                  const sumSamples = findMetric("pipeline_stage_duration_seconds_sum").filter(s => s.labels.stage === stage);
                  const count = countSamples.reduce((s, m) => s + m.value, 0);
                  const total = sumSamples.reduce((s, m) => s + m.value, 0);
                  const avg = count > 0 ? total / count : 0;
                  return (
                    <div key={stage} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
                      <span className="text-sm font-medium text-gray-700 capitalize">{stage.replace(/_/g, " ")}</span>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-gray-500">{count}x</span>
                        <span className="font-mono text-gray-700">{avg.toFixed(1)}s avg</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Model metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Cpu className="size-5" />
                LTR Model
              </CardTitle>
              <CardDescription>Метрики качества LTR-модели</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                {ltrMetrics.map((m) => (
                  <div key={m.labels.metric} className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-500 uppercase">{m.labels.metric}</div>
                    <div className="text-xl font-bold text-gray-900">{m.value.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Pipeline errors detail */}
          {pipelineErrors.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg text-red-700">
                  <AlertCircle className="size-5" />
                  Pipeline errors by stage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {pipelineErrors.map((m, i) => (
                    <div key={i} className="flex items-center justify-between py-1">
                      <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">{m.labels.stage}</Badge>
                      <span className="font-mono text-red-600">{m.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Raw metrics */}
          <details className="text-sm text-gray-500">
            <summary className="cursor-pointer hover:text-gray-700">Сырые метрики</summary>
            <pre className="mt-2 p-4 bg-gray-50 rounded-lg overflow-x-auto text-xs">
              {JSON.stringify(metrics, null, 2)}
            </pre>
          </details>
        </>
      )}
    </div>
  );
}


