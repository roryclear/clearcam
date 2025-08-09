package com.rors.clearcam

import android.Manifest
import android.app.Activity
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.google.firebase.messaging.FirebaseMessaging
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

val Context.dataStore by preferencesDataStore(name = "settings")

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBackPressed: () -> Unit) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val userId = PrefsHelper.getUserId(context) ?: ""
    val notificationsEnabled = remember { mutableStateOf(false) }
    val showLogoutDialog = remember { mutableStateOf(false) }

    // Load saved preference when screen appears
    LaunchedEffect(Unit) {
        context.dataStore.data.map { preferences ->
            preferences[booleanPreferencesKey("notifications_enabled")] ?: false
        }.collect { enabled ->
            notificationsEnabled.value = enabled
        }
    }

    // Logout confirmation dialog
    if (showLogoutDialog.value) {
        AlertDialog(
            onDismissRequest = { showLogoutDialog.value = false },
            title = { Text("Logout") },
            text = { Text("Are you sure you want to logout?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        showLogoutDialog.value = false
                        // Clear user data
                        PrefsHelper.clearUserData(context)
                        // Disable notifications
                        disableNotifications(context)
                        // Restart the app
                        (context as Activity).finish()
                        val intent = context.packageManager
                            .getLaunchIntentForPackage(context.packageName)
                        intent?.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK)
                        context.startActivity(intent)
                    }
                ) {
                    Text("Logout")
                }
            },
            dismissButton = {
                TextButton(
                    onClick = { showLogoutDialog.value = false }
                ) {
                    Text("Cancel")
                }
            }
        )
    }

    BackHandler { onBackPressed() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBackPressed) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Your existing notification toggle - completely unchanged
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Receive event notifications",
                    style = MaterialTheme.typography.bodyLarge
                )

                Switch(
                    checked = notificationsEnabled.value,
                    onCheckedChange = { enabled ->
                        notificationsEnabled.value = enabled
                        coroutineScope.launch {
                            context.dataStore.edit { preferences ->
                                preferences[booleanPreferencesKey("notifications_enabled")] = enabled
                            }
                        }
                        if (enabled) {
                            requestNotificationPermissionAndRegister(context, userId)
                        } else {
                            disableNotifications(context)
                        }
                    }
                )
            }

            // Added logout button at the bottom
            Spacer(modifier = Modifier.weight(1f))
            Button(
                onClick = { showLogoutDialog.value = true },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.error
                )
            ) {
                Text("Logout")
            }
        }
    }
}

private fun disableNotifications(context: Context) {
    FirebaseMessaging.getInstance().deleteToken()
        .addOnCompleteListener { task ->
            if (task.isSuccessful) {
                Log.d("FCM", "FCM token deleted, notifications disabled locally")
            } else {
                Log.e("FCM", "Failed to delete FCM token", task.exception)
            }
        }

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
        notificationManager?.getNotificationChannel("YOUR_CHANNEL_ID")?.let { channel ->
            channel.importance = NotificationManager.IMPORTANCE_NONE
            Log.d("FCM", "Notification channel disabled")
        } ?: Log.e("FCM", "Failed to disable channel (not found or null)")
    }
}

private fun requestNotificationPermissionAndRegister(context: Context, userId: String) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        val hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasPermission) {
            val shouldShowRationale = ActivityCompat.shouldShowRequestPermissionRationale(
                context as Activity,
                Manifest.permission.POST_NOTIFICATIONS
            )

            if (shouldShowRationale) {
                // Could show a dialog here if you want
                Log.d("Settings", "Notification permission rationale should be shown")
                return
            } else {
                ActivityCompat.requestPermissions(
                    context,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    1001
                )
            }
        }
    }

    // Get FCM token and send to server
    FirebaseMessaging.getInstance().token
        .addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                Log.w("FCM", "Fetching FCM token failed", task.exception)
                return@addOnCompleteListener
            }

            val token = task.result
            Log.d("FCM Token", token)

            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val url = URL("https://rors.ai/add_device")
                    val connection = url.openConnection() as HttpURLConnection
                    connection.apply {
                        requestMethod = "POST"
                        setRequestProperty("Content-Type", "application/json")
                        doOutput = true
                        connectTimeout = 10000
                        readTimeout = 10000
                    }

                    val requestBody = """
                        {
                            "device_token": "$token",
                            "session_token": "$userId",
                            "platform": 1
                        }
                    """.trimIndent()

                    OutputStreamWriter(connection.outputStream).use { writer ->
                        writer.write(requestBody)
                    }

                    val responseCode = connection.responseCode
                    if (responseCode == HttpURLConnection.HTTP_OK) {
                        Log.d("FCM", "Token successfully sent to server")
                    } else {
                        Log.e("FCM", "Server returned $responseCode")
                    }
                } catch (e: Exception) {
                    Log.e("FCM", "Failed to send token to server", e)
                }
            }
        }
}
