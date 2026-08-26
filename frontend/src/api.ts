import type {
  GroundTruthRecord,
  ClassDefinition,
  MetricsArtifact,
  PredictionArtifact,
  PresentationSummary,
  RunRecord,
  TimingArtifact,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Preserve the HTTP status when the body is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listRuns: (limit = 100) => request<RunRecord[]>(`/runs?limit=${limit}`),
  getClasses: async () => {
    const result = await request<{fine_classes: ClassDefinition[]}>("/classes");
    return result.fine_classes;
  },
  getRun: (runId: string) => request<RunRecord>(`/runs/${runId}`),
  getPredictions: (runId: string) => request<PredictionArtifact>(`/runs/${runId}/predictions`),
  getMetrics: (runId: string) => request<MetricsArtifact>(`/runs/${runId}/metrics`),
  getTiming: (runId: string) => request<TimingArtifact>(`/runs/${runId}/timing`),
  getPresentation: (runId: string) => request<PresentationSummary>(`/runs/${runId}/presentation`),
  getGroundTruth: async (runId: string) => {
    const result = await request<{ground_truths: GroundTruthRecord[]}>(`/runs/${runId}/ground-truth`);
    return result.ground_truths;
  },
  imageUrl: (runId: string) => `/runs/${runId}/image`,
  uploadImage: async (file: File) => {
    const form = new FormData();
    form.append("image", file);
    return request<RunRecord>("/runs/upload", {method: "POST", body: form});
  },
};
