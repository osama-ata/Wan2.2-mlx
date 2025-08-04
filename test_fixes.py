#!/usr/bin/env python3
"""
Test script to validate critical technical fixes:
1. VAE channel dimensions (16→48 channels)
2. MLX API compatibility
3. Scheduler output handling
4. Simplified VAE wrapper
"""

import sys
import mlx.core as mx
from wan.modules.vae_simple import Wan2_1_VAE, SimpleWanVAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

def test_vae_channel_dimensions():
    """Test VAE with correct 48-channel dimensions"""
    print("🔧 Testing VAE channel dimensions (16→48 channels)...")
    
    # Test SimpleWanVAE
    simple_vae = SimpleWanVAE(z_dim=48)
    assert simple_vae.z_dim == 48, f"Expected z_dim=48, got {simple_vae.z_dim}"
    assert len(simple_vae.mean) == 48, f"Expected 48 mean values, got {len(simple_vae.mean)}"
    assert len(simple_vae.std) == 48, f"Expected 48 std values, got {len(simple_vae.std)}"
    
    # Test Wan2_1_VAE wrapper
    wan_vae = Wan2_1_VAE(z_dim=48)
    assert wan_vae.model.z_dim == 48, f"Expected model z_dim=48, got {wan_vae.model.z_dim}"
    
    print("✅ VAE channel dimensions test passed!")
    return True

def test_vae_wrapper_functionality():
    """Test simplified VAE wrapper functionality"""
    print("🔧 Testing simplified VAE wrapper...")
    
    vae = Wan2_1_VAE(z_dim=48)
    
    # Test image encoding/decoding (4D input)
    test_img = mx.random.normal((3, 256, 256))  # RGB image
    latents = vae.encode([test_img])
    assert len(latents) == 1, "Should return list with one latent"
    assert latents[0].shape[0] == 48, f"Expected 48 channels, got {latents[0].shape[0]}"
    
    # Test video decoding
    decoded = vae.decode(latents)
    assert len(decoded) == 1, "Should return list with one video"
    assert decoded[0].shape[0] == 3, f"Expected 3 RGB channels, got {decoded[0].shape[0]}"
    
    print("✅ VAE wrapper functionality test passed!")
    return True

def test_scheduler_output_handling():
    """Test proper scheduler output handling"""
    print("🔧 Testing scheduler output handling...")
    
    # Test UniPC scheduler
    unipc_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    unipc_scheduler.set_timesteps(10, shift=5.0)
    
    sample = mx.random.normal((1, 48, 5, 16, 16))
    model_output = mx.random.normal((1, 48, 5, 16, 16))
    timestep = unipc_scheduler.timesteps[0]
    
    result = unipc_scheduler.step(model_output, timestep, sample)
    assert hasattr(result, 'prev_sample'), "UniPC scheduler should return SchedulerOutput with prev_sample"
    assert result.prev_sample.shape == sample.shape, "prev_sample should have same shape as input"
    
    # Test DPM++ scheduler  
    dpm_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000)
    dpm_scheduler.set_timesteps(10)
    
    result = dpm_scheduler.step(model_output, timestep, sample)
    assert hasattr(result, 'prev_sample'), "DPM++ scheduler should return SchedulerOutput with prev_sample"
    assert result.prev_sample.shape == sample.shape, "prev_sample should have same shape as input"
    
    print("✅ Scheduler output handling test passed!")
    return True

def test_mlx_compatibility():
    """Test MLX API compatibility"""
    print("🔧 Testing MLX API compatibility...")
    
    # Test basic MLX operations used in the codebase
    test_tensor = mx.random.normal((2, 48, 8, 32, 32))
    
    # Test array operations
    mean_tensor = mx.mean(test_tensor, axis=1, keepdims=True)
    assert mean_tensor.shape == (2, 1, 8, 32, 32), f"Mean operation failed: {mean_tensor.shape}"
    
    # Test repeat operations  
    repeated = mx.repeat(mean_tensor, 3, axis=1)
    assert repeated.shape == (2, 3, 8, 32, 32), f"Repeat operation failed: {repeated.shape}"
    
    # Test expand_dims
    expanded = mx.expand_dims(test_tensor, axis=0)
    assert expanded.shape == (1, 2, 48, 8, 32, 32), f"Expand dims failed: {expanded.shape}"
    
    # Test clip operation
    clipped = mx.clip(test_tensor, -1, 1)
    assert clipped.shape == test_tensor.shape, "Clip operation should preserve shape"
    
    print("✅ MLX API compatibility test passed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Running critical technical fixes validation...\n")
    
    tests = [
        test_vae_channel_dimensions,
        test_vae_wrapper_functionality, 
        test_scheduler_output_handling,
        test_mlx_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All critical technical fixes validated successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
