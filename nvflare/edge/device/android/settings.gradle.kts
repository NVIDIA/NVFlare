pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://oss.sonatype.org/content/repositories/snapshots") }
        maven { url = uri("https://oss.sonatype.org/content/repositories/releases") }
        maven { url = uri("https://jitpack.io") }
        maven { url = uri("https://dl.bintray.com/pytorch/android") }
        maven { url = uri("https://pytorch.org/maven2") }
    }
}

rootProject.name = "nvflare-android-executorch-demo"
include(":app")
