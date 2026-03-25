from jaxtyping import jaxtyped
from beartype import beartype
import jax.numpy as jnp
from jaxtyping_bridge import Float, Array

@jaxtyped
@beartype
def check_shape(x: Float[Array, "3 4"]):
    return x

def test_bridge_catches_mismatch():
    # Correct shape
    x_correct = jnp.zeros((3, 4))
    check_shape(x_correct)
    print("✓ Correct shape passed")

    # Incorrect shape
    x_wrong = jnp.zeros((3, 5))
    try:
        check_shape(x_wrong)
        print("✗ Failed: Bridge did not catch shape mismatch!")
    except Exception as e:
        print(f"✓ Success: Bridge caught shape mismatch as expected")
        # print(f"Error was: {e}")

if __name__ == "__main__":
    test_bridge_catches_mismatch()
