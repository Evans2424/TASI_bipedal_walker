"""Check what environments and dependencies are available."""

import sys
import importlib.util

def check_package(package_name, display_name=None):
    """Check if a package is installed."""
    if display_name is None:
        display_name = package_name

    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: installed (version {version})")
            return True
        except Exception as e:
            print(f"⚠️  {display_name}: installed but error loading: {e}")
            return False
    else:
        print(f"❌ {display_name}: NOT installed")
        return False


def check_environments():
    """Check what Gymnasium environments are available."""
    print("\n" + "="*70)
    print("CHECKING AVAILABLE ENVIRONMENTS")
    print("="*70)

    try:
        import gymnasium as gym

        # Box2D environments
        print("\n🎮 Box2D Environments (Physics-based):")
        box2d_envs = [
            'BipedalWalker-v3',
            'BipedalWalkerHardcore-v3',
            'LunarLander-v2',
            'CarRacing-v2'
        ]

        for env_name in box2d_envs:
            try:
                env = gym.make(env_name)
                env.close()
                print(f"  ✅ {env_name}")
            except Exception as e:
                print(f"  ❌ {env_name}: {str(e)[:50]}")

        # Classic control (always available)
        print("\n🎮 Classic Control (Always Available):")
        classic_envs = [
            'CartPole-v1',
            'MountainCar-v0',
            'Pendulum-v1',
            'Acrobot-v1'
        ]

        for env_name in classic_envs:
            try:
                env = gym.make(env_name)
                env.close()
                print(f"  ✅ {env_name}")
            except Exception as e:
                print(f"  ❌ {env_name}: {str(e)[:50]}")

        # Check for MuJoCo
        print("\n🎮 MuJoCo Environments (Alternative to Box2D):")
        try:
            mujoco_envs = [
                'Humanoid-v5',
                'Walker2d-v5',
                'HalfCheetah-v5',
                'Ant-v5'
            ]

            for env_name in mujoco_envs:
                try:
                    env = gym.make(env_name)
                    env.close()
                    print(f"  ✅ {env_name}")
                except Exception as e:
                    if "mujoco" in str(e).lower():
                        print(f"  ❌ {env_name}: MuJoCo not installed")
                        break
                    else:
                        print(f"  ❌ {env_name}: {str(e)[:50]}")
        except Exception:
            print("  ❌ MuJoCo not available")

    except ImportError:
        print("❌ Gymnasium not installed")


def main():
    """Main entry point."""
    print("="*70)
    print("ENVIRONMENT AVAILABILITY CHECK")
    print("="*70)

    print("\n📦 Checking Python Packages:")
    print("-"*70)

    # Core packages
    check_package("gymnasium", "Gymnasium")
    has_box2d = check_package("Box2D", "Box2D")
    check_package("torch", "PyTorch")
    check_package("numpy", "NumPy")

    # Optional packages
    print("\n📦 Optional Physics Engines:")
    print("-"*70)
    has_mujoco = check_package("mujoco", "MuJoCo")
    has_pybullet = check_package("pybullet", "PyBullet")

    # Check environments
    check_environments()

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if has_box2d:
        print("\n✅ Box2D is installed - You can use BipedalWalker!")
        print("   Your current project will work as-is.")
    else:
        print("\n❌ Box2D is NOT installed - BipedalWalker won't work")
        print("\n🔧 Installation options (no sudo needed):")
        print("   1. pip install gymnasium[box2d] --user")
        print("   2. Use conda: conda install swig && pip install box2d-py")
        print("   3. Build SWIG from source (see documentation)")

        if has_mujoco:
            print("\n✅ You have MuJoCo! You can use alternative environments:")
            print("   - Humanoid-v5 (similar to BipedalWalker)")
            print("   - Walker2d-v5 (2D walking)")
            print("   Modify your config to use these instead.")
        elif has_pybullet:
            print("\n✅ You have PyBullet! Alternative option:")
            print("   - Install: pip install gymnasium-pybullet")
            print("   - Use Walker2DBulletEnv-v0")
        else:
            print("\n⚠️  No physics engines available!")
            print("   You'll need to install Box2D, MuJoCo, or PyBullet")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
