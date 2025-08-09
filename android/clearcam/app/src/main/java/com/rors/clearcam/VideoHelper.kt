package com.rors.clearcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.media.ThumbnailUtils
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.Locale
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// Constants
private const val LIVE_STREAM_TIMEOUT_MS = 15000
private const val AES_KEY_SIZE = 32 // 256-bit key
private const val AES_BLOCK_SIZE = 16
private const val HEADER_SIZE = 8
private val MAGIC_NUMBER = 0x4D41474943UL // "MAGIC"
data class EventVideo(
    val url: String,
    val timestamp: String,
    val date: String,
    val time: String,
    val fileName: String,
    val isDecrypted: Boolean = false
)

fun saveVideoThumbnail(videoPath: String, cameraName: String, context: Context): String? {
    return try {
        val bitmap = getVideoThumbnail(videoPath) ?: return null
        val thumbnailsDir = File(context.filesDir, "thumbnails")
        if (!thumbnailsDir.exists()) {
            thumbnailsDir.mkdirs()
        }

        // Include camera name in the filename
        val thumbnailFile = File(thumbnailsDir, "thumbnail_${cameraName}.jpg")
        thumbnailFile.outputStream().use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
        }
        thumbnailFile.absolutePath
    } catch (e: Exception) {
        null
    }
}

fun getVideoThumbnail(videoPath: String): Bitmap? {
    return try {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(videoPath)

        val duration = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_DURATION
        )?.toLong() ?: 0L
        val midpoint = duration / 2

        val bitmap = retriever.getFrameAtTime(
            midpoint * 1000,
            MediaMetadataRetriever.OPTION_CLOSEST
        )

        retriever.release()

        bitmap?.let {
            ThumbnailUtils.extractThumbnail(
                it,
                320,
                180,
                ThumbnailUtils.OPTIONS_RECYCLE_INPUT
            )
        }
    } catch (e: Exception) {
        null
    }
}

fun getLiveStreamThumbnail(cameraName: String, context: Context): Bitmap? {
    return try {
        val thumbnailsDir = File(context.filesDir, "thumbnails")
        if (!thumbnailsDir.exists()) return null

        val thumbnailFile = File(thumbnailsDir, "thumbnail_${cameraName}.jpg")
        if (thumbnailFile.exists()) {
            thumbnailFile.inputStream().use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } else {
            null
        }
    } catch (e: Exception) {
        null
    }
}

fun parseEventVideos(urls: List<String>): List<EventVideo> {
    return urls.mapNotNull { fullUrl ->
        try {
            val baseUrl = fullUrl.substringBefore("?")
            val filenameWithExt = baseUrl.substringAfterLast("/")
            val filename = filenameWithExt

            // Match all datetime patterns in the filename
            val regex = Regex("""\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}""")
            val allMatches = regex.findAll(filename).toList()

            val lastMatch = allMatches.lastOrNull()?.value ?: return@mapNotNull null

            val (date, time) = lastMatch.split("_")
            val timestamp = "$date $time"
            val displayTime = time.replace("-", ":")

            EventVideo(fullUrl, timestamp, date, displayTime, filename)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }.sortedByDescending { it.timestamp }
}

fun loadAllVideos(videosDir: File): List<EventVideo> {
    return videosDir.listFiles()
        ?.mapNotNull { file: File -> parseFilenameToEventVideo(file.name)
        }
        ?.sortedByDescending { it.timestamp }
        ?: emptyList()
}

fun parseFilenameToEventVideo(filename: String): EventVideo? {
    return try {
        val baseName = if (filename.endsWith(".mp4")) {
            filename.removeSuffix(".mp4")
        } else if (filename.endsWith(".aes")) {
            filename.removeSuffix(".aes")
        } else {
            filename
        }

        // Match all datetime patterns in the filename
        val regex = Regex("""\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}""")
        val allMatches = regex.findAll(baseName).toList()

        // Use the last match (most recent timestamp) if there are multiple
        val lastMatch = allMatches.lastOrNull()?.value ?: return null

        val (date, time) = lastMatch.split("_")
        val timestamp = "$date $time"
        val displayTime = time.replace("-", ":")

        EventVideo(
            url = "",
            timestamp = timestamp,
            date = date,
            time = displayTime,
            fileName = filename,
            isDecrypted = filename.endsWith(".mp4")
        )
    } catch (e: Exception) {
        Log.e("Parse", "Failed to parse filename: $filename", e)
        null
    }
}

suspend fun fetchLiveStreamUrl(userId: String, cameraName: String): String? {
    return withContext(Dispatchers.IO) {
        try {
            val encodedName = URLEncoder.encode(cameraName, "UTF-8")
            val url = URL("https://rors.ai/get_stream_download_link?session_token=$userId&name=$encodedName")

            val connection = url.openConnection() as HttpURLConnection
            connection.apply {
                requestMethod = "GET"
                connectTimeout = LIVE_STREAM_TIMEOUT_MS
                readTimeout = LIVE_STREAM_TIMEOUT_MS
                useCaches = false
            }

            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                connection.inputStream.bufferedReader().use { reader ->
                    val response = reader.readText()
                    JSONObject(response).getString("download_link")
                }
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e("LiveStream", "Failed to fetch stream URL", e)
            null
        }
    }
}

suspend fun deleteStreamLink(userId: String, cameraName: String) {
    withContext(Dispatchers.IO) {
        try {
            val encodedName = URLEncoder.encode(cameraName, "UTF-8")
            val url = URL("https://rors.ai/delete_stream_download_link?session_token=$userId&name=$encodedName")

            val connection = url.openConnection() as HttpURLConnection
            connection.apply {
                requestMethod = "GET"
                connectTimeout = LIVE_STREAM_TIMEOUT_MS
                readTimeout = LIVE_STREAM_TIMEOUT_MS
            }
            connection.connect()
        } catch (e: Exception) {
            Log.e("LiveStream", "Failed to delete stream link", e)
        }
    }
}

suspend fun fetchCameraNames(userId: String): List<String> = withContext(Dispatchers.IO) {
    try {
        val url = URL("https://rors.ai/get_live_devices?session_token=$userId")
        val connection = url.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"

        if (connection.responseCode == 200) {
            val response = connection.inputStream.bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(response)
            val namesArray = jsonObject.getJSONArray("device_names")

            (0 until namesArray.length()).map { i ->
                namesArray.getString(i)
            }
        } else {
            emptyList()
        }
    } catch (e: Exception) {
        e.printStackTrace()
        emptyList()
    }
}

suspend fun fetchEventVideos(
    userId: String,
    newestCreationTime: Long = 0
): List<EventVideo> = withContext(Dispatchers.IO) {
    try {
        val url = URL("https://rors.ai/events?session_token=$userId&newest_creation_time=$newestCreationTime")
        val connection = url.openConnection() as HttpURLConnection
        connection.apply {
            requestMethod = "GET"
            connectTimeout = 15000
            readTimeout = 15000
        }

        if (connection.responseCode == HttpURLConnection.HTTP_OK) {
            val response = connection.inputStream.bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(response)
            val filesArray = jsonObject.getJSONArray("files")

            (0 until filesArray.length()).map { i ->
                filesArray.getString(i)
            }.let { urls ->
                parseEventVideos(urls)
            }
        } else {
            emptyList()
        }
    } catch (e: Exception) {
        Log.e("FetchVideos", "Failed to fetch videos", e)
        emptyList()
    }
}

suspend fun downloadUrl(urlStr: String): ByteArray = withContext(Dispatchers.IO) {
    val url = URL(urlStr)
    val connection = url.openConnection() as HttpURLConnection
    connection.requestMethod = "GET"
    connection.connectTimeout = 15000
    connection.readTimeout = 15000

    connection.inputStream.use { inputStream ->
        inputStream.readBytes()
    }
}

fun getNewestCreationTimeFromFiles(videosDir: File): Long {
    return videosDir.listFiles()
        ?.mapNotNull { file: File ->
            try {
                val baseName = file.name.removeSuffix(".mp4")
                val format = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault())
                format.parse(baseName)?.time ?: 0L
            } catch (e: Exception) {
                null
            }
        }
        ?.maxOrNull() ?: 0L
}

fun decryptAesFileToMp4(
    encryptedInput: ByteArray,
    outputFile: File,
    keyString: String
): Boolean {
    try {
        if (encryptedInput.size <= AES_BLOCK_SIZE) return false

        val iv = encryptedInput.copyOfRange(0, AES_BLOCK_SIZE)
        val encryptedData = encryptedInput.copyOfRange(AES_BLOCK_SIZE, encryptedInput.size)

        val keyBytes = ByteArray(AES_KEY_SIZE).apply {
            val keySrc = keyString.toByteArray(Charsets.UTF_8)
            System.arraycopy(keySrc, 0, this, 0, keySrc.size.coerceAtMost(AES_KEY_SIZE))
        }

        val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")
        val secretKey = SecretKeySpec(keyBytes, "AES")
        val ivSpec = IvParameterSpec(iv)

        cipher.init(Cipher.DECRYPT_MODE, secretKey, ivSpec)
        val decryptedBytes = cipher.doFinal(encryptedData)

        if (decryptedBytes.size < HEADER_SIZE) return false

        val headerValue = (decryptedBytes[0].toULong() and 0xFFu) or
                ((decryptedBytes[1].toULong() and 0xFFu) shl 8) or
                ((decryptedBytes[2].toULong() and 0xFFu) shl 16) or
                ((decryptedBytes[3].toULong() and 0xFFu) shl 24) or
                ((decryptedBytes[4].toULong() and 0xFFu) shl 32) or
                ((decryptedBytes[5].toULong() and 0xFFu) shl 40) or
                ((decryptedBytes[6].toULong() and 0xFFu) shl 48) or
                ((decryptedBytes[7].toULong() and 0xFFu) shl 56)

        if (headerValue != MAGIC_NUMBER) {
            Log.e("Decryption", "Invalid magic number. Expected $MAGIC_NUMBER, got $headerValue")
            return false
        }

        val mp4Data = decryptedBytes.copyOfRange(HEADER_SIZE, decryptedBytes.size)
        outputFile.parentFile?.mkdirs()
        outputFile.writeBytes(mp4Data)

        return true
    } catch (e: Exception) {
        Log.e("Decryption", "Decryption failed", e)
        return false
    }
}