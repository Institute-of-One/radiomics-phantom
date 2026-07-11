"""rphantom -- synthetic phantoms and an IBSI-compliant radiomics core.

An open, fully synthetic research kernel for studying the stability of
radiomics features under acquisition variation.  No patient data, no DICOM, and
no deep learning: every volume is generated from a seed, every feature is
computed from first principles.

Modules
-------
``phantom``
    Deterministic synthetic texture phantoms with known ground truth.
``features``
    IBSI-compliant radiomics feature core (all 11 IBSI families), validated
    against the IBSI digital phantom.
``acquisition``
    Simulated acquisition degradation: blur, noise, resampling, quantisation.
``stability``
    ICC / CCC feature-stability atlas.
``normalize``
    Physics-based feature normalisation.
"""

from __future__ import annotations

from rphantom.acquisition import (
    Acquisition,
    AcquisitionError,
    add_noise,
    apply_blur,
    apply_slice_profile,
    quantise,
    resample,
    simulate_acquisition,
)
from rphantom.features import (
    AGGREGATIONS,
    ZONE_AGGREGATIONS,
    Aggregation,
    Discretisation,
    FeatureError,
    GLCMFeatures,
    GLDZMFeatures,
    GLRLMFeatures,
    GLSZMFeatures,
    IntensityHistogramFeatures,
    IntensityStatistics,
    IVHFeatures,
    LocalIntensityFeatures,
    MorphologyFeatures,
    NGLDMFeatures,
    NGTDMFeatures,
    ZoneAggregation,
    discretise,
    extract_features,
    glcm_features,
    gldzm_features,
    glrlm_features,
    glszm_features,
    intensity_histogram,
    intensity_statistics,
    intensity_volume_histogram,
    local_intensity_features,
    morphology_features,
    ngldm_features,
    ngtdm_features,
)
from rphantom.normalize import (
    CalibrationCurve,
    LinearResponse,
    NormalizationError,
    PowerResponse,
    calibrate_response,
    normalise_feature,
)
from rphantom.phantom import (
    Phantom,
    gaussian_random_field,
    generate_texture_phantom,
    measure_correlation_length,
)
from rphantom.stability import (
    ConcordanceResult,
    FeatureReliability,
    ICCResult,
    StabilityAtlas,
    StabilityError,
    build_stability_atlas,
    concordance_correlation,
    intraclass_correlation,
)

__version__ = "0.6.0"

__all__ = [
    # phantom
    "Phantom",
    "gaussian_random_field",
    "generate_texture_phantom",
    "measure_correlation_length",
    # acquisition
    "Acquisition",
    "AcquisitionError",
    "add_noise",
    "apply_blur",
    "apply_slice_profile",
    "quantise",
    "resample",
    "simulate_acquisition",
    # features
    "AGGREGATIONS",
    "ZONE_AGGREGATIONS",
    "Aggregation",
    "ZoneAggregation",
    "Discretisation",
    "FeatureError",
    "GLCMFeatures",
    "GLDZMFeatures",
    "GLRLMFeatures",
    "GLSZMFeatures",
    "IVHFeatures",
    "IntensityHistogramFeatures",
    "IntensityStatistics",
    "LocalIntensityFeatures",
    "MorphologyFeatures",
    "NGLDMFeatures",
    "NGTDMFeatures",
    "discretise",
    "extract_features",
    "glcm_features",
    "gldzm_features",
    "glrlm_features",
    "glszm_features",
    "intensity_histogram",
    "intensity_statistics",
    "intensity_volume_histogram",
    "local_intensity_features",
    "morphology_features",
    "ngldm_features",
    "ngtdm_features",
    "__version__",
]
