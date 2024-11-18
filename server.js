const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const { v4: uuidv4 } = require("uuid"); // To generate unique IDs

const admin = require("firebase-admin");

const app = express();

const serviceAccount = require("./submissionmlgc-medica-firebase-adminsdk-twauv-c83e4c3deb.json");
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

const storage = multer.memoryStorage();

let model;
const loadModel = async () => {
  console.log("Model is in progress to load...");
  try {
    model = await tf.loadGraphModel(
      "https://storage.googleapis.com/dicoding-submission-medica-2/model.json"
    );
    console.log("Model loaded successfully.");
  } catch (error) {
    console.log("Failed to load model:", error);
  }
};

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 1024 * 1024,
  },
  fileFilter: (req, file, cb) => {
    if (
      file.mimetype === "image/png" ||
      file.mimetype === "image/jpg" ||
      file.mimetype === "image/jpeg"
    ) {
      cb(null, true);
    } else {
      return cb(new Error("Invalid mime type"));
    }
  },
});

const uploadSingleImage = upload.single("image");


app.get("test", function (req, res) {
  return res.json({
    status: true,
    message: "server is work !",
  });
});

app.post("/predict", function (req, res) {
  uploadSingleImage(req, res, async function (err) {
    if (err) {
      if (err.code === "LIMIT_FILE_SIZE") {
        return res.status(413).send({
          status: "fail",
          message:
            "Payload content length greater than maximum allowed: 1000000",
        });
      } else if (err.message === "Invalid mime type") {
        return res.status(400).send({
          status: "fail",
          message:
            "Invalid file type. Only PNG, JPG, or JPEG images are allowed.",
        });
      } else {
        return res.status(500).send({
          status: "error",
          message: "An unexpected error occurred while uploading the file.",
        });
      }
    }

    if (!req.file) {
      return res.status(400).send({
        status: "fail",
        message: "Image is required.",
      });
    }

    const file = req.file;

    const imageBuffer = file.buffer;
    try {
      const tensor = tf.node
        .decodeImage(imageBuffer, 3) // Decode as RGB
        .resizeNearestNeighbor([224, 224]) // Resize to model input size
        .expandDims()
        .div(255.0);

      const prediction = model.predict(tensor);
      const result = prediction.dataSync()[0];
      console.log(result);
      const classification = result > 0.5 ? "Cancer" : "Non-cancer";

      const response = {
        status: "success",
        message: "Model predicted successfully",
        data: {
          id: uuidv4(),
          result: classification,
          suggestion:
            classification === "Cancer"
              ? "Segera periksa ke dokter!"
              : "Penyakit kanker tidak terdeteksi.",
          createdAt: new Date().toISOString(),
        },
      };

      await db
        .collection("predictions")
        .doc(response.data.id)
        .set(response.data);

      return res.status(200).json(response);
    } catch (error) {
      return res.status(400).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }
  });
});

const startServer = async () => {
  try {
    await loadModel(); // Wait for the model to load
    app.listen(3002, () => {
      console.log("Server is listening on port 3002");
    });
  } catch (error) {
    console.log("Failed to start server due to model loading error:", error);
  }
};

startServer();
