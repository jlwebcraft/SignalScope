export interface SpatialEvidence {
  available: boolean;
  heatmap?: string | null;
  attribution_concentration?: number | null;
  target_layer?: string | null;
}

export interface SpectralEvidence {
  available: boolean;
  spectrum?: string | null;
  high_frequency_energy_ratio?: number | null;
  representation?: string;
}

export interface StabilityTransformResult {
  transform_name: string;
  predicted_probability: number;
  delta_from_original: number;
}

export interface RobustnessEvidence {
  available: boolean;
  stability_score: number;
  prediction_flip_rate?: number;
  mean_probability_drift?: number;
  is_stable: boolean;
  degradation_impact: "minimal" | "moderate" | "severe" | string;
  transform_results: StabilityTransformResult[];
}

export interface MetadataEvidence {
  available: boolean;
  has_exif: boolean;
  c2pa_present: boolean;
  camera_make?: string | null;
  camera_model?: string | null;
  software?: string | null;
  anomalies: string[];
}

export interface StructuredEvidence {
  spatial: SpatialEvidence;
  spectral: SpectralEvidence;
  robustness: RobustnessEvidence;
  metadata: MetadataEvidence;
  frequency?: SpectralEvidence;
}

export interface PredictionResponse {
  schema_version: string;
  verdict: "likely_ai_generated" | "likely_real" | "uncertain";
  probability: number;
  raw_probability?: number | null;
  calibrated_probability?: number | null;
  confidence_level: "high" | "medium" | "low" | "uncertain";
  stability_score: number;
  evidence_disagreement: boolean;
  uncertain: boolean;
  is_development_placeholder: boolean;
  evidence: StructuredEvidence;
  explanation: string;
  heatmap_available: boolean;
  disclaimer: string;
}

export interface SystemHealth {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  device: string;
}

export interface SystemInfo {
  name: string;
  description: string;
  hackathon: string;
  schema_version: string;
  modalities: string[];
  supported_formats: string[];
  max_upload_size_mb: number;
  primary_model: string;
  operating_threshold: number;
  uncertainty_band: number;
  ethical_scope: {
    is_identity_system: boolean;
    claims_about_identifiable_people: boolean;
    verdict_types: string[];
    disclaimer: string;
  };
}
