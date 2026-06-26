from dataclasses import dataclass

# Detector geometry in degrees
ADO_THETA_S_DEG = 26.0  # the angle of detectors which are close beam axis
ADO_THETA_L_DEG = 90.0  # the angle of detectors which are far from beam axis

P_THETA_DEG = 90.0  # the angle of polarimeter detectors


# WARNING: variables normally should not be changed casually
L1 = 1
L2 = 2

# Parallel ratio is kept for future extension; currently we are not using it
# KERNEL_RATIO = 0.7

# Numerical safety constants
UNDERFLOW_CUTOFF = -745.0
GAUSSIAN_SIGMA_FLOOR = 1e-6
INV_SQRT_2PI = 0.3989422804014327

# Jacobian diagnostics and output defaults
DETJ_NEAR_ZERO_THRESHOLD = 1e-12
DENSITY_REL_THRESHOLDS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
TERMINAL_DENSITY_THRESHOLD = 1e-3
CSV_COMPRESS_LEVEL = 1


@dataclass(frozen=True)
class OutputConfig:
    compress_csv: bool = True
    csv_compress_level: int = CSV_COMPRESS_LEVEL
    export_detj_split: bool = True
    export_detj_regular: bool = True
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD


@dataclass(frozen=True)
class GridConfig:
    mode: str = "full"  # "full" or "test"

    # full mode (mirrors notebook)
    theta_min_full: float = -89.0
    theta_max_full: float = 89.0
    theta_step_full: float = 0.01

    # When Jacobian check report status is warning, one needs to check the detJ = 0 graph of singular file to avoid complete alignment for some sigma/I
    # Lower limit: for integer 0.15/Ji; for half-integer 0.19/Ji
    sigma_i_min_full: float = 0.039 
    sigma_i_max_full: float = 1.00
    sigma_i_step_full: float = 0.001

    # test mode
    theta_min_test: float = -45.0
    theta_max_test: float = 45.0
    theta_step_test: float = 0.2

    sigma_i_min_test: float = 0.10
    sigma_i_max_test: float = 0.60
    sigma_i_step_test: float = 0.02

    def theta_range(self):
        if self.mode == "test":
            return self.theta_min_test, self.theta_max_test, self.theta_step_test
        return self.theta_min_full, self.theta_max_full, self.theta_step_full

    def sigma_i_range(self):
        if self.mode == "test":
            return (
                self.sigma_i_min_test,
                self.sigma_i_max_test,
                self.sigma_i_step_test,
            )
        return self.sigma_i_min_full, self.sigma_i_max_full, self.sigma_i_step_full


@dataclass(frozen=True)
class PdfConfig:
    skew_symmetry_tolerance: float = 1e-8
    skew_fit_quality_cutoff: float = 2.0e-2
    skew_shape_bound: float = 50.0
    skew_scale_factor_upper: float = 10.0
    skew_center_shift_factor: float = 5.0
    skew_accept_shape_soft_limit: float = 15.0
    skew_accept_scale_min_frac: float = 0.05
    skew_accept_near_bound_frac: float = 0.02
    skew_accept_q16_tol_frac: float = 0.08
    skew_accept_q50_tol_frac: float = 0.08
    skew_accept_q84_tol_frac: float = 0.08
    skew_reject_q16_tol_frac: float = 0.20
    skew_reject_q50_tol_frac: float = 0.20
    skew_reject_q84_tol_frac: float = 0.20
    verbose_pdf_fit: bool = False
    use_split_normal_fallback: bool = True
