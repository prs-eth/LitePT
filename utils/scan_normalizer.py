class ScanNormalizer:
    """
    Applies (or reverses) a configured sequence of affine transforms to
    intra-oral scans.

    Parameters
    ----------
    config_path : str | Path
        Path to a YAML configuration file specifying the transform pipelines
        for the upper and/or lower scans.

    Usage
    -----
    normalizer = ScanNormalizer("config.yaml")

    # Forward: apply transforms in the order listed in the config
    mesh_out   = normalizer.apply(mesh_in,   jaw="upper")
    points_out = normalizer.apply(points_in, jaw="lower")

    # Inverse: apply transforms in reverse order with inverse matrices
    mesh_inv   = normalizer.apply_inverse(mesh_in,   jaw="upper")
    points_inv = normalizer.apply_inverse(points_in, jaw="lower")
    """

    JAW_KEYS = ("upper", "lower")

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = None if config_path is None else Path(config_path)
        self._matrices: dict[str, list[np.ndarray]] = {
            "upper": [],
            "lower": [],
        }
        if self.config_path is not None:
            self._load_config()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        with open(self.config_path, "r") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ValueError("Config file must be a YAML mapping.")

        for jaw in self.JAW_KEYS:
            jaw_cfg = raw.get(jaw)
            if jaw_cfg is None:
                continue  # jaw not specified → empty pipeline
            transforms_cfg = jaw_cfg.get("transforms", [])
            matrices = [_parse_transform(t) for t in transforms_cfg]
            self._matrices[jaw] = matrices

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matrices(
        self, jaw: Literal["upper", "lower"]
    ) -> list[np.ndarray]:
        """
        Return a *copy* of the list of 4×4 transform matrices for *jaw*
        in forward order.
        """
        self._check_jaw(jaw)
        return [m.copy() for m in self._matrices[jaw]]

    def apply(
        self,
        scan: trimesh.Trimesh | np.ndarray,
        jaw: Literal["upper", "lower"],
        inplace: bool = False,
    ) -> trimesh.Trimesh | np.ndarray:
        """
        Apply the configured transforms in **forward** order.

        Parameters
        ----------
        scan : trimesh.Trimesh or np.ndarray (N, 3)
            The scan to transform.
        jaw : "upper" | "lower"
        inplace : bool, default False
            If True and *scan* is a trimesh.Trimesh, mutate it directly and
            return the same object. Has no effect on numpy arrays, which are
            always returned as new arrays.

        Returns
        -------
        Transformed object of the same type as *scan*.
        """
        self._check_jaw(jaw)
        return self._apply_sequence(scan, self._matrices[jaw], inplace=inplace)

    def apply_inverse(
        self,
        scan: trimesh.Trimesh | np.ndarray,
        jaw: Literal["upper", "lower"],
        inplace: bool = False,
    ) -> trimesh.Trimesh | np.ndarray:
        """
        Apply the configured transforms in **reverse** order, each inverted.

        This undoes the forward pipeline:
            apply_inverse(apply(scan)) ≈ scan

        Parameters
        ----------
        scan : trimesh.Trimesh or np.ndarray (N, 3)
            The scan to transform.
        jaw : "upper" | "lower"
        inplace : bool, default False
            If True and *scan* is a trimesh.Trimesh, mutate it directly and
            return the same object. Has no effect on numpy arrays, which are
            always returned as new arrays.

        Returns
        -------
        Transformed object of the same type as *scan*.
        """
        self._check_jaw(jaw)
        inv_matrices = [
            np.linalg.inv(m) for m in reversed(self._matrices[jaw])
        ]
        return self._apply_sequence(scan, inv_matrices, inplace=inplace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_jaw(jaw: str) -> None:
        if jaw not in ScanNormalizer.JAW_KEYS:
            raise ValueError(
                f"jaw must be one of {ScanNormalizer.JAW_KEYS}, got '{jaw}'."
            )

    @staticmethod
    def _apply_sequence(
        scan: trimesh.Trimesh | np.ndarray,
        matrices: list[np.ndarray],
        inplace: bool = False,
    ) -> trimesh.Trimesh | np.ndarray:
        """Apply a list of 4×4 matrices sequentially to *scan*."""
        if isinstance(scan, trimesh.Trimesh):
            return ScanNormalizer._apply_to_mesh(scan, matrices, inplace=inplace)
        elif isinstance(scan, np.ndarray):
            return ScanNormalizer._apply_to_points(scan, matrices)
        else:
            raise TypeError(
                f"scan must be a trimesh.Trimesh or np.ndarray, "
                f"got {type(scan).__name__}."
            )

    @staticmethod
    def _apply_to_mesh(
        mesh: trimesh.Trimesh, matrices: list[np.ndarray], inplace: bool = False,
    ) -> trimesh.Trimesh:
        """Return a transformed mesh, either mutated in place or as a deep copy."""
        result = mesh if inplace else copy.deepcopy(mesh)
        for mat in matrices:
            result.apply_transform(mat)
        return result

    @staticmethod
    def _apply_to_points(
        points: np.ndarray, matrices: list[np.ndarray]
    ) -> np.ndarray:
        """
        Apply a sequence of 4×4 matrices to an (N, 3) point array.
        Returns a new (N, 3) float64 array; the input is never mutated.
        """
        pts = np.array(points, dtype=float)
        if pts.ndim == 1 and pts.shape[0] == 3:
            pts = pts[np.newaxis, :]   # handle a single point gracefully
        elif pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"points must have shape (N, 3), got {pts.shape}."
            )

        # Homogeneous coordinates: (N, 4)
        ones = np.ones((pts.shape[0], 1), dtype=float)
        hom = np.concatenate([pts, ones], axis=1)

        for mat in matrices:
            hom = (mat @ hom.T).T   # (4,4) @ (4,N) → (4,N) → (N,4)

        # Perspective divide (safe for affine mats where w≈1, but robust)
        result = hom[:, :3] / hom[:, 3:4]
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        upper_n = len(self._matrices["upper"])
        lower_n = len(self._matrices["lower"])
        return (
            f"ScanNormalizer(config='{self.config_path}', "
            f"upper_transforms={upper_n}, lower_transforms={lower_n})"
        )