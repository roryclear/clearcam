package com.rors.clearcam
import androidx.compose.foundation.gestures.detectTapGestures
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import android.widget.VideoView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import kotlinx.coroutines.*
import java.io.File
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Pause
import androidx.activity.compose.BackHandler
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.Manifest
import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import com.google.firebase.messaging.FirebaseMessaging
import com.rors.clearcam.SettingsScreen
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.io.FileOutputStream
import java.net.URLDecoder
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.geometry.Offset


private const val APP_VIDEO_DIR = "videos"
private const val SEGMENT_DOWNLOAD_INTERVAL_MS = 1000L
private const val LINK_REFRESH_INTERVAL_MS = 50000L

@Composable
private fun DecryptionKeyDialog(
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit
) {
    var decryptionKey by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Enter Decryption Key") },
        text = {
            Column {
                Text("This video is encrypted. Please enter the decryption key:")
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = decryptionKey,
                    onValueChange = { decryptionKey = it },
                    label = { Text("Decryption Key") },
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Password,
                        autoCorrect = false
                    ),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onConfirm(decryptionKey) },
                enabled = decryptionKey.isNotBlank()
            ) {
                Text("Decrypt")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

object ThumbnailCache {
    private const val CACHE_DIR = "thumbnails"

    fun getCachedThumbnail(context: Context, videoId: String): Bitmap? {
        val cacheFile = getCacheFile(context, videoId)
        return if (cacheFile.exists()) {
            BitmapFactory.decodeFile(cacheFile.absolutePath)
        } else {
            null
        }
    }

    fun cacheThumbnail(context: Context, videoId: String, bitmap: Bitmap) {
        val cacheFile = getCacheFile(context, videoId)
        try {
            FileOutputStream(cacheFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun deleteCachedThumbnail(context: Context, videoId: String) {
        val cacheFile = getCacheFile(context, videoId)
        if (cacheFile.exists()) {
            cacheFile.delete()
        }
    }

    private fun getCacheFile(context: Context, videoId: String): File {
        val cacheDir = File(context.cacheDir, CACHE_DIR)
        if (!cacheDir.exists()) cacheDir.mkdirs()
        return File(cacheDir, "thumbnail_${videoId.hashCode()}.jpg")
    }
}


suspend fun deleteVideoFromBackend(userId: String, filename: String) {
    withContext(Dispatchers.IO) {
        try {
            val backendFilename = filename
            val encodedName = URLEncoder.encode(backendFilename, "UTF-8")
            val url = URL("https://rors.ai/video?session_token=$userId&name=$encodedName")

            val connection = url.openConnection() as HttpURLConnection
            connection.apply {
                requestMethod = "DELETE"
                connectTimeout = 15000
                readTimeout = 15000
            }

            connection.connect()
            connection.responseCode
        } catch (e: Exception) {
            Log.e("DeleteVideo", "Failed to delete video from backend", e)
        }
    }
}

fun getDecryptionKeys(context: Context): MutableList<String> {
    val prefs = context.getSharedPreferences("clearcam_prefs", Context.MODE_PRIVATE)
    val keysString = prefs.getString("decryption_keys", null)
    return keysString?.split("||")?.toMutableList() ?: mutableListOf()
}

fun saveDecryptionKeys(context: Context, keys: List<String>) {
    val prefs = context.getSharedPreferences("clearcam_prefs", Context.MODE_PRIVATE)
    prefs.edit().putString("decryption_keys", keys.joinToString("||")).apply()
}

fun addDecryptionKey(context: Context, key: String) {
    val keys = getDecryptionKeys(context)
    if (key !in keys) {
        keys.add(key)
        saveDecryptionKeys(context, keys)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(userId: String) {
    var cameraDevices by remember { mutableStateOf<List<Map<String, Any>>>(emptyList()) }
    var eventVideos by remember { mutableStateOf<List<EventVideo>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showVideoPlayer by remember { mutableStateOf(false) }
    var currentVideoPath by remember { mutableStateOf("") }
    var showLiveStream by remember { mutableStateOf(false) }
    var currentCameraName by remember { mutableStateOf("") }
    var showSettings by remember { mutableStateOf(false) }
    var isOnline by remember { mutableStateOf(true) } // Add online status state

    if (showSettings) {
        SettingsScreen(onBackPressed = { showSettings = false })
        return
    }

    val context = LocalContext.current

    val coroutineScope = rememberCoroutineScope()
    val videosDir = remember { File(context.filesDir, APP_VIDEO_DIR) }

    val scrollState = rememberScrollState()
    var isRefreshing by remember { mutableStateOf(false) }
    var refreshTrigger by remember { mutableStateOf(0f) }

    // Function to check internet status
    fun checkInternetStatus() {
        coroutineScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val url = URL("https://rors.ai/ping")
                    val connection = url.openConnection() as HttpURLConnection
                    connection.apply {
                        requestMethod = "GET"
                        connectTimeout = 10000
                        readTimeout = 10000
                    }
                    connection.connect()
                    // If we get here without exception, we're online
                    isOnline = true
                }
            } catch (e: Exception) {
                isOnline = false
            }
        }
    }

    // Auto-refresh internet status every 10 seconds
    LaunchedEffect(userId) {
        while (true) {
            checkInternetStatus()
            delay(10000L)
        }
    }

    fun loadVideosFromStorage() {
        eventVideos = loadAllVideos(videosDir)
    }

    fun refreshData() {
        coroutineScope.launch {
            try {
                if (!videosDir.exists()) videosDir.mkdirs()
                val camerasDeferred = async { fetchCameraJson(userId) }
                val newestTime = getNewestCreationTimeFromFiles(videosDir)
                val newVideos = fetchEventVideos(userId, newestTime / 1000)
                val camerasJson = camerasDeferred.await()
                cameraDevices = if (camerasJson != null) {
                    val devicesArray = camerasJson.optJSONArray("devices")
                    if (devicesArray != null) {
                        (0 until devicesArray.length()).map { i ->
                            val deviceObj = devicesArray.getJSONObject(i)
                            mapOf(
                                "name" to deviceObj.optString("name", "Unknown"),
                                "alerts_on" to deviceObj.optInt("alerts_on", 0)
                            )
                        }
                    } else {
                        emptyList()
                    }
                } else {
                    emptyList()
                }
                val decryptionKeys = getDecryptionKeys(context).toMutableList()
                videosDir.listFiles()?.forEach { file ->
                    if (file.name.endsWith(".aes")) {
                        val outputFilename = file.name.removeSuffix(".aes")
                        val videoFile = File(videosDir, outputFilename)

                        if (!videoFile.exists()) {
                            for (key in decryptionKeys) {
                                try {
                                    val aesData = file.readBytes()
                                    if (decryptAesFileToMp4(aesData, videoFile, key)) {
                                        file.delete()
                                        break
                                    }
                                } catch (e: Exception) {
                                    Log.e("Decrypt", "Failed to decrypt ${file.name} with key $key", e)
                                }
                            }
                        }
                    }
                }

                newVideos.forEach { video ->
                    try {
                        val outputFilename = video.fileName.removeSuffix(".aes")
                        val videoFile = File(videosDir, outputFilename)
                        val encryptedFile = File(videosDir, video.fileName)

                        if (!videoFile.exists()) {
                            val aesData = downloadUrl(video.url)
                            encryptedFile.writeBytes(aesData)

                            var success = false
                            for (key in decryptionKeys) {
                                if (decryptAesFileToMp4(aesData, videoFile, key)) {
                                    success = true
                                    encryptedFile.delete()
                                    break
                                }
                            }

                            if (!success) {
                                videoFile.delete()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("Decrypt", "Failed for ${video.fileName}", e)
                    }
                }

                loadVideosFromStorage()
                errorMessage = null
            } catch (e: Exception) {
                errorMessage = "Error: ${e.message}"
            } finally {
                isLoading = false
            }
        }
    }

    LaunchedEffect(userId) {
        refreshData()
        checkInternetStatus() // Check status on initial load
    }

    LaunchedEffect(userId) {
        while (true) {
            delay(10000L)
            refreshData()
            checkInternetStatus() // Check status with each refresh
        }
    }

    if (showVideoPlayer) {
        SimpleVideoPlayer(
            videoPath = currentVideoPath,
            onDismiss = { showVideoPlayer = false }
        )
        return
    }

    if (showLiveStream) {
        LiveStreamPlayer(
            userId = userId,
            cameraName = currentCameraName,
            onDismiss = { showLiveStream = false }
        )
        return
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectVerticalDragGestures { _, dragAmount ->
                    if (scrollState.value == 0 && dragAmount > 0) {
                        refreshTrigger = dragAmount.coerceAtMost(150f)
                        if (refreshTrigger >= 150f && !isRefreshing) {
                            isRefreshing = true
                            refreshData()
                            isRefreshing = false
                            refreshTrigger = 0f
                        }
                    }
                }
            }
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(scrollState)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Spacer(modifier = Modifier.width(48.dp))

                IconButton(onClick = { showSettings = true }) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = "Settings"
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Online/Offline Status Indicator - Centered
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center // Changed to Center
            ) {
                // Status dot
                Box(
                    modifier = Modifier
                        .size(12.dp)
                        .clip(CircleShape)
                        .background(
                            if (isOnline) Color(0xFF34C759) // iOS systemGreen equivalent
                            else Color(0xFFFF3B30) // iOS systemRed equivalent
                        )
                )

                Spacer(modifier = Modifier.width(8.dp))

                // Status text
                Text(
                    text = if (isOnline) "online" else "offline",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF8E8E93) // iOS systemGray equivalent
                )
            }

            if (refreshTrigger > 0) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(64.dp),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(36.dp),
                        strokeWidth = 3.dp
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            when {
                isLoading -> Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }

                errorMessage != null -> Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = errorMessage!!,
                        color = MaterialTheme.colorScheme.error,
                        textAlign = TextAlign.Center
                    )
                }

                else -> {
                    CameraHorizontalList(
                        cameraDevices = cameraDevices,
                        userId = userId,
                        onCameraClick = { cameraName ->
                            currentCameraName = cameraName
                            showLiveStream = true
                        },
                        onRefresh = {
                            refreshData()
                            checkInternetStatus()
                        }
                    )

                    Spacer(modifier = Modifier.height(24.dp))

                    EventVideoSection(
                        eventVideos = eventVideos,
                        context = context,
                        onVideoClick = { videoPath ->
                            currentVideoPath = videoPath
                            showVideoPlayer = true
                        },
                        onVideoDeleted = { deletedVideo ->
                            val file = File(videosDir, deletedVideo.fileName)
                            if (file.exists()) {
                                file.delete()
                            }
                            ThumbnailCache.deleteCachedThumbnail(context, deletedVideo.fileName)

                            coroutineScope.launch {
                                deleteVideoFromBackend(userId, deletedVideo.fileName)
                                refreshData()
                            }
                            eventVideos = eventVideos.filter { it != deletedVideo }
                        },
                        onVideoShared = { video ->
                            val videoFile = File(videosDir, video.fileName)
                            val uri = FileProvider.getUriForFile(
                                context,
                                "${context.packageName}.provider",
                                videoFile
                            )
                            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                                type = "video/mp4"
                                putExtra(Intent.EXTRA_STREAM, uri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            context.startActivity(Intent.createChooser(shareIntent, "Share Video"))
                        },
                        userId = userId
                    )
                }
            }
        }
    }
}

@Composable
fun SimpleVideoPlayer(videoPath: String, onDismiss: () -> Unit) {
    val context = LocalContext.current
    var showError by remember { mutableStateOf(false) }
    var isPlaying by remember { mutableStateOf(true) }
    var currentPosition by remember { mutableStateOf(0) }
    var duration by remember { mutableStateOf(0) }
    var showControls by remember { mutableStateOf(false) }
    var isScrubbing by remember { mutableStateOf(false) }
    val videoViewRef = remember { mutableStateOf<VideoView?>(null) }
    val coroutineScope = rememberCoroutineScope()

    var isPrepared by remember { mutableStateOf(false) }
    var isInitialPlay by remember { mutableStateOf(true) }

    LaunchedEffect(videoPath) {
        videoViewRef.value?.apply {
            showError = false
            isPrepared = false

            stopPlayback()
            setVideoPath("")

            setVideoPath(videoPath)
        }
    }

    fun stopAndDismiss() {
        videoViewRef.value?.stopPlayback()
        onDismiss()
    }

    BackHandler(enabled = true) {
        stopAndDismiss()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .clickable { showControls = !showControls }
    ) {
        AndroidView(
            factory = { ctx ->
                VideoView(ctx).apply {
                    videoViewRef.value = this
                    setVideoPath(videoPath)

                    setOnPreparedListener { mp ->
                        duration = mp.duration
                        isPrepared = true
                        showError = false

                        if (isInitialPlay) {
                            mp.start()
                            isInitialPlay = false
                        }
                        isPlaying = true

                        coroutineScope.launch {
                            while (isActive) {
                                if (!isScrubbing) {
                                    currentPosition = this@apply.currentPosition
                                }
                                delay(200)
                            }
                        }
                    }

                    setOnCompletionListener {
                        isPlaying = false
                        currentPosition = duration
                        // Auto-replay
                        seekTo(0)
                        start()
                    }

                    setOnErrorListener { _, what, extra ->
                        if (isPrepared) {
                            showError = true
                        }
                        true
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        if (showError) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.7f))
                    .clickable { showError = false },
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text(
                        text = "Error playing video",
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium
                    )
                    Button(
                        onClick = {
                            showError = false
                            videoViewRef.value?.start()
                        }
                    ) {
                        Text("Retry")
                    }
                }
            }
        }

        if (showControls && !showError) {
            Box(modifier = Modifier.fillMaxSize()) {
                // Top controls
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                        .align(Alignment.TopStart),
                    horizontalArrangement = Arrangement.Start
                ) {
                    IconButton(
                        onClick = { stopAndDismiss() },
                        modifier = Modifier
                            .size(36.dp)
                            .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = Color.White
                        )
                    }
                }

                // Center play/pause button
                IconButton(
                    onClick = {
                        isPlaying = !isPlaying
                        videoViewRef.value?.apply {
                            if (isPlaying) start() else pause()
                        }
                    },
                    modifier = Modifier
                        .size(64.dp)
                        .align(Alignment.Center)
                        .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                ) {
                    Icon(
                        imageVector = if (isPlaying) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                        contentDescription = if (isPlaying) "Pause" else "Play",
                        tint = Color.White,
                        modifier = Modifier.size(36.dp)
                    )
                }

                // Bottom controls
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.BottomCenter)
                        .padding(16.dp)
                ) {
                    Slider(
                        value = currentPosition.toFloat(),
                        onValueChange = { newValue ->
                            isScrubbing = true
                            currentPosition = newValue.toInt()
                        },
                        onValueChangeFinished = {
                            isScrubbing = false
                            videoViewRef.value?.seekTo(currentPosition)
                        },
                        valueRange = 0f..duration.toFloat(),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 8.dp),
                        colors = SliderDefaults.colors(
                            thumbColor = Color.White,
                            activeTrackColor = Color.Red,
                            inactiveTrackColor = Color.White.copy(alpha = 0.3f)
                        )
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = formatMillisToTime(currentPosition),
                            color = Color.White,
                            style = MaterialTheme.typography.labelSmall
                        )
                        Text(
                            text = formatMillisToTime(duration),
                            color = Color.White,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }
            }
        }
    }
}

private fun formatMillisToTime(millis: Int): String {
    val seconds = (millis / 1000) % 60
    val minutes = (millis / (1000 * 60)) % 60
    return String.format("%02d:%02d", minutes, seconds)
}

@Composable
fun LiveStreamPlayer(
    userId: String,
    cameraName: String,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val videoViewRef = remember { mutableStateOf<VideoView?>(null) }
    val coroutineScope = rememberCoroutineScope()
    var currentSegmentPath by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var showError by remember { mutableStateOf(false) }

    val decryptionKeys = remember { mutableStateListOf<String>() }
    var showDecryptionDialog by remember { mutableStateOf(false) }
    var pendingSegmentData by remember { mutableStateOf<ByteArray?>(null) }

    fun captureThumbnail() {
        currentSegmentPath?.let { path ->
            coroutineScope.launch(Dispatchers.IO) {
                saveVideoThumbnail(path, cameraName, context)
            }
        }
    }

    // Proper back button handling
    BackHandler(enabled = true) {
        videoViewRef.value?.stopPlayback()
        onDismiss()
    }

    // Load saved decryption keys
    LaunchedEffect(Unit) {
        decryptionKeys.clear()
        decryptionKeys.addAll(getDecryptionKeys(context))
    }

    // Stream fetching logic
    DisposableEffect(userId, cameraName) {
        val coroutineScope = CoroutineScope(Dispatchers.IO)
        var downloadLink: String? = null
        var isPlaying = false

        val linkRefreshJob = coroutineScope.launch {
            while (true) {
                try {
                    downloadLink = fetchLiveStreamUrl(userId, cameraName)
                } catch (e: Exception) {
                    Log.e("LiveStream", "Error fetching link", e)
                }
                delay(LINK_REFRESH_INTERVAL_MS)
            }
        }

        val segmentDownloadJob = coroutineScope.launch {
            while (true) {
                try {
                    val link = downloadLink
                    if (link != null) {
                        val segmentData = try {
                            downloadUrl(link)
                        } catch (e: Exception) {
                            null
                        }

                        segmentData?.let { data ->
                            val tempFile = File.createTempFile(
                                "segment_${System.currentTimeMillis()}",
                                ".mp4",
                                context.cacheDir
                            ).apply { deleteOnExit() }

                            var success = false

                            for (storedKey in decryptionKeys) {
                                if (decryptAesFileToMp4(data, tempFile, storedKey)) {
                                    success = true
                                    break
                                }
                            }

                            if (success) {
                                currentSegmentPath = tempFile.absolutePath
                                isLoading = false

                                withContext(Dispatchers.Main) {
                                    videoViewRef.value?.apply {
                                        if (!isPlaying) {
                                            setVideoPath(tempFile.absolutePath)
                                            setOnPreparedListener { mp ->
                                                mp.start()
                                                isPlaying = true
                                                captureThumbnail()
                                                isLoading = false
                                            }
                                        } else {
                                            setOnCompletionListener {
                                                setVideoPath(tempFile.absolutePath)
                                                setOnPreparedListener { mp ->
                                                    mp.start()
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                pendingSegmentData = data
                                showDecryptionDialog = true
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("LiveStream", "Failed to process segment", e)
                    isLoading = false
                    showError = true
                }

                delay(SEGMENT_DOWNLOAD_INTERVAL_MS)
            }
        }

        onDispose {
            linkRefreshJob.cancel()
            segmentDownloadJob.cancel()
            videoViewRef.value?.stopPlayback()
        }
    }

    if (showDecryptionDialog) {
        DecryptionKeyDialog(
            onDismiss = {
                showDecryptionDialog = false
                pendingSegmentData = null
            },
            onConfirm = { userKey ->
                showDecryptionDialog = false
                pendingSegmentData?.let { data ->
                    val tempFile = File.createTempFile(
                        "segment_user_${System.currentTimeMillis()}",
                        ".mp4",
                        context.cacheDir
                    ).apply { deleteOnExit() }

                    coroutineScope.launch {
                        if (decryptAesFileToMp4(data, tempFile, userKey)) {
                            addDecryptionKey(context, userKey)
                            decryptionKeys.add(userKey)
                            currentSegmentPath = tempFile.absolutePath
                            isLoading = false

                            withContext(Dispatchers.Main) {
                                videoViewRef.value?.apply {
                                    setVideoPath(tempFile.absolutePath)
                                    setOnPreparedListener { mp ->
                                        mp.start()
                                        captureThumbnail()
                                    }
                                }
                            }
                        } else {
                            withContext(Dispatchers.Main) {
                                showError = true
                                Toast.makeText(context, "Invalid decryption key", Toast.LENGTH_SHORT).show()
                            }
                        }
                        pendingSegmentData = null
                    }
                }
            }
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        AndroidView(
            factory = { ctx ->
                VideoView(ctx).also { videoView ->
                    videoViewRef.value = videoView
                    videoView.setOnErrorListener { _, _, _ ->
                        showError = true
                        true
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        if (isLoading) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.7f)),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(48.dp),
                    color = Color.White,
                    strokeWidth = 4.dp
                )
            }
        }

        if (showError) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.7f))
                    .clickable { showError = false },
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text(
                        text = "Error loading stream",
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium
                    )
                    Button(
                        onClick = {
                            showError = false
                            isLoading = true
                            videoViewRef.value?.start()
                        }
                    ) {
                        Text("Retry")
                    }
                }
            }
        }

        IconButton(
            onClick = {
                videoViewRef.value?.stopPlayback()
                onDismiss()
            },
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Close,
                contentDescription = "Close",
                tint = Color.White
            )
        }
    }
}

@Composable
fun CameraHorizontalList(
    cameraDevices: List<Map<String, Any>>,
    userId: String,
    onCameraClick: (String) -> Unit,
    onRefresh: () -> Unit
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Your Cameras",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            IconButton(
                onClick = onRefresh,
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Refresh,
                    contentDescription = "Refresh cameras"
                )
            }
        }

        if (cameraDevices.isEmpty()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "No live cameras found",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                Button(
                    onClick = onRefresh,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary
                    )
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Refresh",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Refresh")
                }
            }
        } else {
            Row(
                modifier = Modifier
                    .horizontalScroll(rememberScrollState())
                    .padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                cameraDevices.forEach { device ->
                    val cameraName = device["name"] as? String ?: "Unknown"
                    val alertsOn = device["alerts_on"] as? Int ?: 0
                    CameraCard(
                        name = cameraName,
                        alertsOn = alertsOn,
                        userId = userId,
                        onClick = { onCameraClick(cameraName) }
                    )
                }
            }
        }
    }
}

@Composable
fun CameraCard(
    name: String,
    alertsOn: Int,
    userId: String, // Add userId parameter
    onClick: () -> Unit
) {
    val context = LocalContext.current
    var thumbnailBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var currentAlertsState by remember { mutableStateOf(alertsOn) }
    var isLoading by remember { mutableStateOf(false) }
    val coroutineScope = rememberCoroutineScope()

    // Update local state when props change
    LaunchedEffect(alertsOn) {
        currentAlertsState = alertsOn
    }

    LaunchedEffect(name) {
        thumbnailBitmap = withContext(Dispatchers.IO) {
            getLiveStreamThumbnail(name, context)
        }
    }

    suspend fun toggleAlerts(newState: Int) {
        if (isLoading) return

        isLoading = true
        try {
            val success = withContext(Dispatchers.IO) {
                try {
                    val url = URL("https://rors.ai/toggle_alerts")
                    val connection = url.openConnection() as HttpURLConnection
                    connection.apply {
                        requestMethod = "POST"
                        setRequestProperty("Content-Type", "application/json")
                        doOutput = true
                        connectTimeout = 15000
                        readTimeout = 15000
                    }

                    val encodedDeviceName = URLEncoder.encode(name, "UTF-8")

                    val body = """
                    {
                        "session_token": "$userId",
                        "device_name": "$encodedDeviceName",
                        "alerts_on": $newState
                    }
                """.trimIndent()

                    OutputStreamWriter(connection.outputStream).use { writer ->
                        writer.write(body)
                        writer.flush()
                    }

                    val responseCode = connection.responseCode
                    responseCode == HttpURLConnection.HTTP_OK
                } catch (e: Exception) {
                    false
                }
            }

            if (success) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Alerts ${if (newState == 1) "enabled" else "disabled"}", Toast.LENGTH_SHORT).show()
                }
            } else {
                currentAlertsState = alertsOn
                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "Failed to update alerts", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            currentAlertsState = alertsOn
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Network error", Toast.LENGTH_SHORT).show()
            }
        } finally {
            isLoading = false
        }
    }

    Box(
        modifier = Modifier
            .width(280.dp)
            .aspectRatio(16f / 9f)
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
    ) {
        thumbnailBitmap?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Live stream thumbnail",
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize()
            )
        } ?: run {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.verticalGradient(
                            colors = listOf(Color(0xFF3A3A3A), Color(0xFF1A1A1A))
                        )
                    )
            )
        }

        Icon(
            imageVector = Icons.Default.PlayArrow,
            contentDescription = "Play",
            tint = Color.White,
            modifier = Modifier
                .size(48.dp)
                .align(Alignment.Center)
        )

        // Camera name and toggle switch
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = URLDecoder.decode(name, StandardCharsets.UTF_8.name()),
                color = Color.White,
                style = MaterialTheme.typography.titleMedium
            )

            // Alerts label and toggle switch
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = Color.White
                    )
                } else {
                    Text(
                        text = "Alerts",
                        color = Color.White,
                        style = MaterialTheme.typography.labelMedium
                    )
                }
                Switch(
                    checked = currentAlertsState == 1,
                    onCheckedChange = { newState ->
                        val newAlertsState = if (newState) 1 else 0
                        currentAlertsState = newAlertsState
                        coroutineScope.launch {
                            toggleAlerts(newAlertsState)
                        }
                    },
                    enabled = !isLoading
                )
            }
        }
    }
}

@Composable
fun EventVideoSection(
    eventVideos: List<EventVideo>,
    context: Context,
    onVideoClick: (String) -> Unit,
    onVideoDeleted: (EventVideo) -> Unit,
    onVideoShared: (EventVideo) -> Unit,
    userId: String
) {
    if (eventVideos.isEmpty()) {
        Text(
            text = "No events found",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(16.dp)
        )
        return
    }

    val videosDir = File(context.filesDir, APP_VIDEO_DIR)
    val groupedByDate = eventVideos.groupBy { it.date }
        .toList()
        .sortedByDescending { (date, _) -> date }

    val coroutineScope = rememberCoroutineScope()

    fun handleDecryption(video: EventVideo, key: String) {
        coroutineScope.launch {
            try {
                val encryptedFile = File(videosDir, video.fileName)
                val outputFile = File(videosDir, video.fileName.removeSuffix(".aes") + ".mp4")

                if (decryptAesFileToMp4(encryptedFile.readBytes(), outputFile, key)) {
                    addDecryptionKey(context, key)
                    encryptedFile.delete()
                    ThumbnailCache.deleteCachedThumbnail(context, video.fileName)
                    onVideoClick(outputFile.absolutePath)
                } else {
                    if (outputFile.exists()) outputFile.delete()
                    Toast.makeText(context, "Invalid decryption key", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e("Decryption", "Failed to decrypt video", e)
                Toast.makeText(context, "Decryption failed", Toast.LENGTH_SHORT).show()
            }
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(
            text = "Events",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        groupedByDate.forEach { (date, videosOnDate) ->
            Column {
                Text(
                    text = date,
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(vertical = 8.dp)
                )

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    videosOnDate.forEach { video ->
                        key(video.fileName) {
                            EventVideoListItem(
                                video = video,
                                videoPath = File(videosDir, video.fileName).absolutePath,
                                onClick = {
                                    onVideoClick(File(videosDir, video.fileName).absolutePath)
                                },
                                onDelete = { onVideoDeleted(video) },
                                onShare = { onVideoShared(video) },
                                onDecrypt = { key -> handleDecryption(video, key) }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun EventVideoListItem(
    video: EventVideo,
    videoPath: String,
    onClick: () -> Unit,
    onDelete: () -> Unit,
    onShare: () -> Unit,
    onDecrypt: (String) -> Unit
) {
    val context = LocalContext.current
    var thumbnailBitmap by remember { mutableStateOf<Bitmap?>(null) }
    val cacheKey = remember { video.fileName }
    var showMenu by remember { mutableStateOf(false) }
    var showDecryptionDialog by remember { mutableStateOf(false) }
    var isActive by remember { mutableStateOf(true) }

    // Load thumbnail in a coroutine
    LaunchedEffect(video.fileName) {
        isActive = true

        if (video.isDecrypted) {
            val bitmap = ThumbnailCache.getCachedThumbnail(context, cacheKey)
                ?: withContext(Dispatchers.IO) {
                    if (isActive) {
                        getVideoThumbnail(videoPath)?.also {
                            ThumbnailCache.cacheThumbnail(context, cacheKey, it)
                        }
                    } else null
                }

            if (isActive) {
                thumbnailBitmap = bitmap
            }
        } else {
            ThumbnailCache.deleteCachedThumbnail(context, cacheKey)
        }
    }

    // Cleanup when the composable is disposed
    DisposableEffect(video.fileName) {
        onDispose {
            isActive = false
            thumbnailBitmap = null
        }
    }

    if (showDecryptionDialog) {
        DecryptionKeyDialog(
            onDismiss = { showDecryptionDialog = false },
            onConfirm = { key ->
                showDecryptionDialog = false
                onDecrypt(key)
            }
        )
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable {
                if (video.isDecrypted) {
                    onClick()
                } else {
                    showDecryptionDialog = true
                }
            }
            .pointerInput(Unit) {
                detectTapGestures(
                    onTap = {
                        // Regular tap - play video
                        if (video.isDecrypted) {
                            onClick()
                        } else {
                            showDecryptionDialog = true
                        }
                    },
                    onLongPress = {
                        // Long press - save thumbnail and open with system viewer
                        if (video.isDecrypted && thumbnailBitmap != null) {
                            // Save thumbnail to temp file and open with system viewer
                            val tempFile = File.createTempFile("thumbnail_", ".jpg", context.cacheDir)
                            tempFile.outputStream().use { stream ->
                                thumbnailBitmap!!.compress(Bitmap.CompressFormat.JPEG, 90, stream)
                            }

                            val uri = FileProvider.getUriForFile(
                                context,
                                "${context.packageName}.provider",
                                tempFile
                            )

                            val intent = Intent(Intent.ACTION_VIEW).apply {
                                setDataAndType(uri, "image/jpeg")
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }

                            try {
                                context.startActivity(intent)
                            } catch (e: Exception) {
                                Toast.makeText(context, "No image viewer app found", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                )
            }
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Thumbnail (160x90)
        Box(
            modifier = Modifier
                .width(160.dp)
                .aspectRatio(16f / 9f)
                .clip(RoundedCornerShape(8.dp))
        ) {
            if (thumbnailBitmap != null) {
                Image(
                    bitmap = thumbnailBitmap!!.asImageBitmap(),
                    contentDescription = "Video thumbnail",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            brush = Brush.verticalGradient(
                                colors = listOf(Color(0xFF3A3A3A), Color(0xFF1A1A1A))
                            )
                        )
                )
            }

            Icon(
                imageVector = if (video.isDecrypted) Icons.Default.PlayArrow else Icons.Default.Lock,
                contentDescription = if (video.isDecrypted) "Play" else "Encrypted",
                tint = if (video.isDecrypted) Color.White.copy(alpha = 0.8f) else Color.Red.copy(alpha = 0.8f),
                modifier = Modifier
                    .size(24.dp)
                    .align(Alignment.Center)
            )
        }

        // Video info and menu
        Row(
            modifier = Modifier
                .weight(1f)
                .padding(start = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = video.time,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = video.date,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            // Context menu
            Box {
                IconButton(
                    onClick = { showMenu = true },
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.MoreVert,
                        contentDescription = "More options",
                        tint = MaterialTheme.colorScheme.onSurface
                    )
                }

                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Share") },
                        onClick = {
                            showMenu = false
                            onShare()
                        },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.Share,
                                contentDescription = "Share"
                            )
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Delete", color = MaterialTheme.colorScheme.error) },
                        onClick = {
                            showMenu = false
                            onDelete()
                        },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "Delete",
                                tint = MaterialTheme.colorScheme.error
                            )
                        }
                    )
                }
            }
        }
    }
}