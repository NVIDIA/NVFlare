#!/usr/bin/env python3
"""
FedHCA2 NVFLARE Experiment Runner
Clean implementation that preserves original FedHCA2 algorithms
"""

import argparse
import os
import sys

from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner


def create_fedhca2_job(num_rounds=10):
    """Create the FedHCA2 federated learning job"""

    job_name = "fedhca2_pascal_nyud"

    # Create FedJob
    job = FedJob(name=job_name, min_clients=6)

    print(f"🚀 Creating FedHCA2 NVFLARE Job: {job_name}")
    print(f"   🔄 Rounds: {num_rounds}")
    print(f"   👥 Minimum clients: 6")

    # Add FedHCA2 aggregator to server
    print(f"   🤖 Adding FedHCA2Aggregator to server...")
    from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
    from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

    # Import custom components
    job_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(job_dir, "jobs", "fedhca2_pascal_nyud", "app", "custom"))

    from fedhca2_aggregator import FedHCA2Aggregator
    from fedhca2_learner import FedHCA2Learner

    # Server components
    aggregator = FedHCA2Aggregator(
        encoder_agg="conflict_averse",
        decoder_agg="cross_attention",
        ca_c=0.4,
        enc_alpha_init=0.1,
        dec_beta_init=0.1,
        hyperweight_lr=0.01,
    )

    shareable_generator = FullModelShareableGenerator()

    controller = ScatterAndGather(
        min_clients=6,
        num_rounds=num_rounds,
        start_round=0,
        wait_time_after_min_received=10,
        aggregator_id="aggregator",
        shareable_generator_id="shareable_generator",
        train_task_name="train",
        train_timeout=3600,
    )

    job.to_server(aggregator, "aggregator")
    job.to_server(shareable_generator, "shareable_generator")
    job.to_server(controller)

    print(f"   ✅ Server components configured")

    # Client configurations
    print(f"   👥 Configuring heterogeneous clients...")

    # Pascal Context Single-Task Clients (site-1 to site-5)
    pascal_tasks = ["semseg", "human_parts", "normals", "edge", "sal"]
    task_names = {
        "semseg": "Semantic Segmentation",
        "human_parts": "Human Body Parts",
        "normals": "Surface Normals",
        "edge": "Edge Detection",
        "sal": "Saliency Detection",
    }

    for i, task in enumerate(pascal_tasks):
        site_name = f"site-{i+1}"
        print(f"      🎯 {site_name}: {task_names[task]} ({task})")

        learner = FedHCA2Learner(
            lr=0.0001,
            weight_decay=0.0001,
            local_epochs=1,
            batch_size=4,
            warmup_epochs=5,
            fp16=True,
            backbone_type="swin-t",
            backbone_pretrained=True,
        )

        job.to(learner, site_name)

    # NYU Depth Multi-Task Client (site-6)
    print(f"      🎯 site-6: Multi-Task (semseg + normals + edge + depth)")

    nyud_learner = FedHCA2Learner(
        lr=0.0001,
        weight_decay=0.0001,
        local_epochs=1,  # TODO:set to 4
        batch_size=4,
        warmup_epochs=5,
        fp16=True,
        backbone_type="swin-t",
        backbone_pretrained=True,
    )

    job.to(nyud_learner, "site-6")

    print(f"   ✅ All clients configured")
    print(f"      📊 5 Pascal Context single-task clients")
    print(f"      📊 1 NYU Depth multi-task client")

    return job


def main():
    parser = argparse.ArgumentParser(description="Run FedHCA2 NVFLARE Experiment")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--workspace", type=str, default="/tmp/fedhca2_nvflare", help="Workspace directory")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

    args = parser.parse_args()

    print("🌟" * 80)
    print("🚀 FEDHCA2 NVFLARE EXPERIMENT")
    print("🌟" * 80)
    print(f"⚙️  Configuration:")
    print(f"   🔄 Rounds: {args.rounds}")
    print(f"   💾 Workspace: {args.workspace}")
    print(f"   🖥️  GPU: {args.gpu}")
    print()

    print("🎯 FedHCA2 Features:")
    print("   ✅ Conflict-Averse Encoder Aggregation")
    print("   ✅ Cross-Attention Decoder Aggregation")
    print("   ✅ Learnable Hyperweights for Personalization")
    print("   ✅ Heterogeneous Single-Task + Multi-Task Clients")
    print("   ✅ Exact Original Algorithm Implementation")
    print()

    # Create job
    print("🏗️  Creating FedHCA2 job...")
    job = create_fedhca2_job(num_rounds=args.rounds)

    # Export job
    export_dir = os.path.join(args.workspace, "jobs")
    print(f"\n📤 Exporting job to {export_dir}...")
    job.export_job(export_dir)
    print(f"   ✅ Job exported successfully")

    # Run simulation
    workdir = os.path.join(args.workspace, "workdir")
    print(f"\n🚀 Starting NVFLARE simulation...")
    print(f"   📁 Workspace: {workdir}")
    print(f"   🖥️  GPU: {args.gpu}")
    print()

    try:
        job.simulator_run(workdir, gpu=args.gpu)

        print("\n" + "🎉" * 60)
        print("🏆 FEDHCA2 EXPERIMENT COMPLETED!")
        print("🎉" * 60)
        print(f"📊 Results:")
        print(f"   📁 Workspace: {workdir}")
        print(f"   📈 Logs: {workdir}/server/simulate_job/")
        print(f"   📊 Client logs: {workdir}/site-*/simulate_job/")
        print()
        print("📈 Monitor training:")
        print(f"   tensorboard --logdir={workdir}/server/simulate_job/tb_events")

    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED")
        print("🔥" * 60)
        print(f"💥 Error: {e}")
        print("🔍 Check logs for details:")
        print(f"   📁 Workspace: {workdir}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

