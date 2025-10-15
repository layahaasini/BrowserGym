#!/usr/bin/env python3
"""
Simple BrowserGym Agent Test
============================

This script demonstrates how to test a web agent in BrowserGym.
Run this to understand the basic evaluation process before building your green agent.

Usage:
    python simple_agent_test.py
"""

import gymnasium as gym
import browsergym.core  # register the openended task
import time

def test_basic_environment():
    """Test basic BrowserGym environment setup and interaction."""
    
    print("🚀 Starting BrowserGym Agent Test")
    print("=" * 50)
    
    # Create an openended environment
    print("📝 Creating BrowserGym environment...")
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": "https://www.google.com"},
        headless=False,  # Show browser so you can see what's happening
        wait_for_user_message=True,  # Wait for user input
        max_episode_steps=10,  # Limit steps for demo
    )
    
    # Reset environment
    print("🔄 Resetting environment...")
    obs, info = env.reset()
    
    print("\n📊 Initial Environment State:")
    print(f"  - Observation keys: {list(obs.keys())}")
    print(f"  - Goal: {info.get('goal', 'No specific goal')}")
    print(f"  - Available pages: {obs.get('open_pages_urls', [])}")
    
    # Show what the agent can observe
    print("\n👀 Agent Observation:")
    if 'axtree_txt' in obs:
        print(f"  - Accessibility tree (first 200 chars): {obs['axtree_txt'][:200]}...")
    if 'chat_messages' in obs:
        print(f"  - Chat messages: {obs['chat_messages']}")
    
    # Test a simple action
    print("\n🎯 Testing agent action: take_screenshot()")
    action = "take_screenshot()"
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")
    print(f"  - Info keys: {list(info.keys())}")
    
    # Test navigation action
    print("\n🎯 Testing agent action: navigate to a different page")
    action = "navigate('https://www.github.com')"
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  - Reward: {reward}")
    print(f"  - Current URL: {obs.get('open_pages_urls', ['Unknown'])[0]}")
    
    # Close environment
    print("\n🔚 Closing environment...")
    env.close()
    
    print("\n✅ Test completed successfully!")
    print("\n💡 Key Takeaways:")
    print("  - BrowserGym provides rich observations (HTML, accessibility tree, screenshots)")
    print("  - Agents can take various actions (navigate, click, type, etc.)")
    print("  - Environment returns rewards and status information")
    print("  - This is the foundation for evaluating web agents!")

def test_miniwob_task():
    """Test a simple MiniWoB task to understand benchmark evaluation."""
    
    print("\n" + "=" * 50)
    print("🎮 Testing MiniWoB Benchmark Task")
    print("=" * 50)
    
    try:
        # Import MiniWoB
        import browsergym.miniwob
        
        # Create a simple MiniWoB task
        print("📝 Creating MiniWoB environment...")
        env = gym.make(
            "browsergym/miniwob.click-dialog",
            headless=False,
            max_episode_steps=5,  # Short demo
        )
        
        # Reset environment
        print("🔄 Resetting MiniWoB environment...")
        obs, info = env.reset()
        
        print(f"  - Task: {info.get('task_name', 'Unknown')}")
        print(f"  - Goal: {info.get('goal', 'No specific goal')}")
        print(f"  - Observation keys: {list(obs.keys())}")
        
        # Try a simple action
        print("\n🎯 Testing action: click on first available element")
        if 'axtree_txt' in obs and 'click' in obs['axtree_txt'].lower():
            action = "click('1')"  # Try clicking element with bid 1
        else:
            action = "take_screenshot()"  # Fallback action
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  - Reward: {reward}")
        print(f"  - Success: {info.get('success', False)}")
        print(f"  - Terminated: {terminated}")
        
        env.close()
        print("✅ MiniWoB test completed!")
        
    except ImportError:
        print("❌ MiniWoB not available. Install with: pip install browsergym-miniwob")
    except Exception as e:
        print(f"❌ MiniWoB test failed: {e}")

def main():
    """Run all tests to demonstrate BrowserGym capabilities."""
    
    print("🎯 BrowserGym Agent Testing Suite")
    print("This will help you understand how web agent evaluation works")
    print("=" * 60)
    
    # Test basic environment
    test_basic_environment()
    
    # Test MiniWoB benchmark
    test_miniwob_task()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📚 Next Steps:")
    print("  1. Study the demo_agent/agent.py to see a full agent implementation")
    print("  2. Run: python demo_agent/run_demo.py --task_name openended --start_url https://www.google.com")
    print("  3. Experiment with different benchmarks and tasks")
    print("  4. Start building your green agent evaluation system!")
    print("\n💡 Remember: Your green agent will orchestrate multiple white agents")
    print("   running these same types of evaluations in parallel!")

if __name__ == "__main__":
    main()
