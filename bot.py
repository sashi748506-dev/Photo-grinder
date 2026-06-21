import os
import cv2
import numpy as np
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ─── Logging ───
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Image Processing (same pipeline) ───
def grind_and_smooth(image_path: str, output_path: str) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Extra smoothness
    smooth = img.copy()
    for _ in range(5):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # Color grind (posterization)
    Z = smooth.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    K = 7
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape((img.shape))

    # Ink edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 9, 2)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edged = cv2.multiply(quantized, edge_mask, scale=1/255.0)

    # Warm color grading
    hsv = cv2.cvtColor(edged, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    b, g, r = cv2.split(graded)
    r = cv2.add(r, 18)
    g = cv2.add(g, 8)
    b = cv2.subtract(b, 12)
    graded = cv2.merge([b, g, r])

    # Final contrast
    lab = cv2.cvtColor(graded, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    final = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


# ─── Telegram Handlers ───
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 Send me any photo and I'll *grind its colors* with *extra smoothness*.\n"
        "Everything runs locally — no external AI APIs.",
        parse_mode="Markdown"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Grinding colors... ⏳")
    tmp_in = tmp_out = None

    try:
        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)

        tmp_in = f"tmp_in_{update.message.message_id}.jpg"
        tmp_out = f"tmp_out_{update.message.message_id}.jpg"

        await tg_file.download_to_drive(tmp_in)

        if grind_and_smooth(tmp_in, tmp_out):
            await update.message.reply_photo(photo=open(tmp_out, "rb"))
        else:
            await update.message.reply_text("❌ Couldn't process that image.")

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("⚠️ Processing failed. Try again?")
    finally:
        await msg.delete()
        for f in (tmp_in, tmp_out):
            if f and os.path.exists(f):
                os.remove(f)


def main():
    TOKEN = os.environ.get("BOT_TOKEN")
    if not TOKEN:
        raise ValueError("BOT_TOKEN environment variable not set!")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("🤖 Bot started polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
