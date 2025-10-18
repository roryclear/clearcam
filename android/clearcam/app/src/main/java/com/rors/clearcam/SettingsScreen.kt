package com.rors.clearcam

import android.Manifest
import android.app.Activity
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import android.widget.Toast
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

                val activity = LocalContext.current as? Activity // todo

                Switch(
                    checked = notificationsEnabled.value,
                    onCheckedChange = { enabled ->
                        if (enabled) {
                            // Temporarily disable switch until confirmation
                            notificationsEnabled.value = false

                            requestNotificationPermissionAndRegister(activity, context, userId) { success ->
                                // Must switch back to main thread for UI update
                                coroutineScope.launch(Dispatchers.Main) {
                                    if (success) {
                                        notificationsEnabled.value = true
                                        context.dataStore.edit { prefs ->
                                            prefs[booleanPreferencesKey("notifications_enabled")] = true
                                        }
                                    } else {
                                        notificationsEnabled.value = false
                                        Toast.makeText(context, "No connection. Could not enable notifications.", Toast.LENGTH_SHORT).show()
                                    }
                                }
                            }
                        } else {
                            notificationsEnabled.value = false
                            coroutineScope.launch {
                                context.dataStore.edit { prefs ->
                                    prefs[booleanPreferencesKey("notifications_enabled")] = false
                                }
                            }
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

private fun requestNotificationPermissionAndRegister(
    activity: Activity?,
    context: Context,
    userId: String,
    onResult: (Boolean) -> Unit
) {
    // Step 1: permission check (only if activity is available)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        val hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasPermission) {
            if (activity != null) {
                ActivityCompat.requestPermissions(
                    activity,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    1001
                )
            }
            onResult(false)
            return
        }
    }

    // Step 2: Get FCM token safely
    FirebaseMessaging.getInstance().token
        .addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                onResult(false)
                return@addOnCompleteListener
            }

            val token = task.result
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
                    onResult(responseCode == HttpURLConnection.HTTP_OK)
                } catch (e: Exception) {
                    onResult(false)
                }
            }
        }
}

