import jax
import jax.numpy as jnp


class CalibrationGeometry:
    def __init__(self, Y: jax.Array, retain_rank: int | None = None, tol: float = 1e-6):
        """
        Constructs an affine geometry (subspace) from a set of calibration activations Y.
        Args:
            Y: (N, D) array of unperturbed calibration activations.
            retain_rank: Optional number of top singular vectors to retain. If None,
                        retains all singular vectors with value > tol.
            tol: Tolerance for non-zero singular values.
        """
        self.mu = jnp.mean(Y, axis=0)
        Y_c = Y - self.mu

        # Compute SVD of centered activations
        # Y_c = U * S * Vh
        U, S, Vh = jnp.linalg.svd(Y_c, full_matrices=False)

        # Determine valid rank
        valid_mask = S > tol
        valid_rank = jnp.sum(valid_mask).item()

        if retain_rank is not None:
            self.k = min(valid_rank, retain_rank)
        else:
            self.k = valid_rank

        # Vh is (min(N, D), D). We want the first k rows (which are the principal directions).
        # We store V_k of shape (D, k)
        self.V_k = Vh[: self.k, :].T
        self.S = S
        self.tol = tol
        self.retain_rank = retain_rank

    def distance(self, y: jax.Array) -> jax.Array:
        """
        Computes the Euclidean distance from a point y to the affine subspace G(Y).
        Args:
            y: (..., D) array of query points.
        Returns:
            distances: (...) array of distances.
        """
        # Center y
        y_c = y - self.mu

        # Project y_c onto the subspace spanned by V_k
        # p = y_c @ V_k @ V_k.T
        coeffs = jnp.dot(y_c, self.V_k)  # (..., k)
        p = jnp.dot(coeffs, self.V_k.T)  # (..., D)

        # Residual
        r = y_c - p

        # Distance is norm of residual
        return jnp.linalg.norm(r, axis=-1)
