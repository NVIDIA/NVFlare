Job meta.json: Launcher Configuration Design Options
Background
NVFlare jobs use meta.json to describe how a job should be launched at each site. The process, Docker, Kubernetes,
and Slurm modes have different execution settings, while portable resources such as `num_of_gpus` apply across
launchers.
The core design question is whether meta.json should separate scheduler-facing resource requirements from launcher-specific execution configuration, or combine them into a single structure.
This distinction matters because the server scheduler wants a clean, launcher-agnostic view of required resources for placement decisions, while each launcher needs concrete, mode-specific settings to actually run the job.
The schema must support explicit job images for Docker, Kubernetes, and Slurm. Default-image policy is separate from
that schema choice.
Note: the current code reads Docker, Kubernetes, and Slurm job images from `launcher_spec`; older
`deploy_map`-based behavior is no longer the primary path.

Implemented outcome: Option 4 is the current schema. `launcher_spec["default"][mode]` is shallow-merged with
`launcher_spec[site][mode]`, with the site block winning. `resource_spec[site].num_of_gpus` remains portable. The
Slurm block accepts image and job-owned resource/time settings; site policy stays outside job metadata.

Option 1: Launcher namespaces inside resource_spec
Each site entry in resource_spec contains a sub-key for each launcher mode. A launcher reads only its own block.
{
  "deploy_map": { "app": ["@ALL"] },
  "resource_spec": {
    "site-1": {
      "k8s": { "image": "repo/nvflare:2.7.2", "num_of_gpus": 2 },
      "docker": { "image": "repo/nvflare:2.7.2", "shm_size": "8g", "num_of_gpus": 2 },
      "process": { "num_of_gpus": 2 }
    }
  }
}
Pros:
Launcher-specific configuration is explicit and isolated
A launcher reads only the fields that belong to it
New launcher modes can be added without colliding with existing mode-specific fields
Cons:
resource_spec is meant to represent resources, so fields like image and shm_size do not fit its semantics
Shared values such as num_of_gpus must be repeated across launcher blocks unless another fallback rule is introduced
It is verbose for simple jobs that use the same image or same GPU setting across modes

Option 2: Flat universal args in resource_spec
Each site entry uses a single flat dictionary. All launchers see the same keys, and each launcher ignores the ones that do not apply.
{
  "deploy_map": { "app": ["@ALL"] },
  "resource_spec": {
    "site-1": {
      "image": "repo/nvflare:2.7.2",
      "shm_size": "8g",
      "num_of_gpus": 2,
      "mem_per_gpu_in_GiB": 16
    }
  }
}
Pros:
Simplest format to write and read
No need to know which launcher will be used at authoring time
Backward compatible with existing flat resource_spec jobs
All job-related requirements appear in one place
Cons:
Launcher-specific keys become noise for launchers that do not use them
Docker-only and K8s-only fields share one namespace, which increases the chance of misuse or confusion
The schema is forced toward a lowest-common-denominator model, which makes launcher-specific evolution harder
Validation becomes harder because the same key space has different meanings for different launchers

Option 3: Launcher config mixed into deploy_map
In this model, deploy_map entries carry both deployment targeting and launcher configuration.
{
  "deploy_map": {
    "app": [
      {
        "sites": ["@ALL"],
        "image": "repo/nvflare:2.7.2"
      }
    ]
  }
}
The structure could be extended further if needed:
{
  "deploy_map": {
    "app": [
      {
        "sites": ["site-1"],
        "image": "repo/nvflare:2.7.2",
        "shm_size": "8g"
      },
      {
        "sites": ["site-2"],
        "image": "repo/nvflare:2.7.3"
      }
    ]
  }
}
Pros:
Keeps image selection close to deployment targeting, so it is easy to see which image goes to which site
Feels natural if the team views image selection as part of deployment mapping
Can express per-site image overrides without introducing another top-level section
Cons:
deploy_map is conceptually about routing apps to sites, not about how jobs are launched, so the concerns become mixed
It does not scale cleanly as more launcher settings are added, such as Docker shm_size or K8s CPU and memory limits
It becomes harder to define what belongs in deploy_map versus what belongs elsewhere
It risks turning deploy_map into a catch-all structure instead of keeping it a clean deployment map

Option 4: Two top-level sections: resource_spec and launcher_spec
This option introduces two explicit top-level keys with separate responsibilities:
resource_spec: abstract resource requirements, used by the server scheduler for placement and by launchers for hardware-related configuration. This section stays launcher-agnostic.
launcher_spec: launcher-specific execution configuration, namespaced by launcher mode. The scheduler ignores this section entirely.
{
  "deploy_map": { "app": ["@ALL"] },
  "resource_spec": {
    "site-1": { "num_of_gpus": 2, "mem_per_gpu_in_GiB": 16 }
  },
  "launcher_spec": {
    "site-1": {
      "k8s": { "image": "repo/nvflare:2.7.2", "cpu": "500m", "memory": "2Gi", "ephemeral_storage": "4Gi" },
      "docker": { "image": "repo/nvflare:2.7.2", "shm_size": "8g" },
      "slurm": { "nodes": 2, "gpus_per_node": 1, "cpus_per_node": 8, "mem_per_node": 32768, "time": "02:00:00" }
    }
  }
}
In this model, num_of_gpus remains in resource_spec as the scheduler-facing requirement. Docker and K8s translate the flat resource_spec[site] value into their runtime-specific GPU request:
Docker: --gpus=2
K8s: resources.limits["nvidia.com/gpu"] = 2
Slurm: one-node GPU request by default, or `--gres=gpu:<gpus_per_node>` with validated multi-node topology
Process: passed as an environment variable or argument
Launcher-only fields such as image, shm_size, K8s CPU/memory/ephemeral storage, and Slurm topology/time values use launcher_spec as their canonical location. The existing nested resource_spec compatibility fallback is also accepted when launcher_spec has no matching site/mode block.
This makes image a first-class launcher field for both Docker and K8s, instead of an implicit value derived from unrelated configuration.
Pros:
Separation of concerns is explicit and unambiguous
resource_spec stays scheduler-facing, while launcher_spec stays launcher-facing
Shared values such as num_of_gpus are defined once and translated by each launcher without duplication
New launcher modes can be added by introducing a new block in launcher_spec
Validation is cleaner because the scheduler and launcher validate different inputs
Cons:
Introduces a new top-level key, which is a schema change
Existing jobs already use flat resource_spec, so backward compatibility needs to be considered
Job authors must consider two sections instead of one
Implemented extension: default in launcher_spec
Option 4 supports a default block to reduce repetition for common launcher settings such as image, time, or pending timeout.
{
  "deploy_map": { "app": ["@ALL"] },
  "resource_spec": {
    "site-1": { "num_of_gpus": 2, "mem_per_gpu_in_GiB": 16 }
  },
  "launcher_spec": {
    "default": {
      "k8s": { "image": "repo/nvflare:2.7.2" },
      "docker": { "image": "repo/nvflare:2.7.2" },
      "slurm": { "time": "02:00:00", "pending_timeout": 300 }
    },
    "site-1": {
      "k8s": { "cpu": "500m", "memory": "2Gi" },
      "docker": { "shm_size": "8g" },
      "slurm": { "nodes": 2, "gpus_per_node": 1 }
    }
  }
}
Merge rule:
Start with default[mode]
Overlay site[mode]
Site-specific values win
This reduces verbosity for jobs that use the same launcher image or other settings everywhere. The merge is shallow,
and `default` is a reserved key rather than a site name.

Comparison


Option 1
Option 2
Option 3
Option 4
Launcher isolation
✓
✗
✗
✓
No repeated fields
✗
✓
~
✓
Clear separation of concerns
✗
~
✗
✓
Backward compatible
✓
✓
✗
~
Scales to new launchers
✓
✗
✗
✓
Handles common defaults well
✗
✗
~
~ (+✓ with extension)

Option 4 gets ~ on backward compatibility because it introduces launcher_spec, but it can still preserve compatibility during migration by continuing to support the current resource_spec contract.

Recommendation
Recommended: Option 4
Option 4 is the only option that cleanly separates what resources are required from how the job is launched. That separation gives it the strongest long-term foundation.
Its main advantages are:
Clear semantics: resource_spec is scheduler input, launcher_spec is launcher input
No required duplication of shared resource fields such as num_of_gpus because launchers read the flat resource_spec value
Better extensibility as new launcher modes or launcher-specific settings are added
Cleaner validation boundaries between scheduler logic and launcher logic
Although it introduces a new top-level key, it can still be rolled out in a backward-compatible way by continuing to support the current flat resource_spec format during migration.
The default extension is implemented as a shallow merge without changing the core separation of concerns.
Pragmatic fallback: Option 1
If the immediate priority is to minimize implementation change, Option 1 is a reasonable interim step. However, it keeps launcher-specific configuration inside resource_spec, which weakens semantics and is less clean as a long-term model.

Resolved and remaining questions
Who is the authoritative consumer of resource_spec: the server scheduler, the launcher, or both? This determines whether GPU-related fields must remain there or could move elsewhere.
GPU counts should remain in flat resource_spec so the scheduler and launchers read the same resource requirement.
Should image be required explicitly for Docker and K8s in launcher_spec, or may a launcher fall back to the CP/SP image when image is absent?
Is backward compatibility with flat resource_spec a hard requirement, or would a migration guide be sufficient?
The default extension uses only shallow per-launcher merging; recursive merging is not supported.

The implemented merge is shallow. Job images use the existing BYOC authorization. Backend-specific validation and
image precedence are documented with each launcher.
