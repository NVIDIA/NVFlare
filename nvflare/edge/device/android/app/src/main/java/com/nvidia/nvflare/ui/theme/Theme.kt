package com.nvidia.nvflare.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = NVFlareBlue,
    onPrimary = SystemBackground,
    primaryContainer = NVFlareLightBlue,
    onPrimaryContainer = NVFlareDarkBlue,
    secondary = NVFlareDarkBlue,
    onSecondary = SystemBackground,
    background = SystemBackground,
    onBackground = SystemText,
    surface = SystemBackground,
    onSurface = SystemText,
    error = Error,
    onError = SystemBackground
)

private val DarkColorScheme = darkColorScheme(
    primary = NVFlareLightBlue,
    onPrimary = SystemBackgroundDark,
    primaryContainer = NVFlareDarkBlue,
    onPrimaryContainer = NVFlareLightBlue,
    secondary = NVFlareLightBlue,
    onSecondary = SystemBackgroundDark,
    background = SystemBackgroundDark,
    onBackground = SystemTextDark,
    surface = SystemBackgroundDark,
    onSurface = SystemTextDark,
    error = Error,
    onError = SystemBackgroundDark
)

@Composable
fun NVFlareTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
} 