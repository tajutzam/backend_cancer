const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const { v4: uuidv4 } = require("uuid"); // To generate unique IDs

const admin = require("firebase-admin");

const app = express();

const serviceAccount = require("./submissionmlgc-danaariska-4a0dace443a3.json");
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
      "https://storage.googleapis.com/dicoding-danaariska/submissions-model/model.json"
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

app.get("/predict/histories", async (req, res) => {
  try {
    const snapshot = await db.collection("predictions").get();
    const predictions = snapshot.docs.map((doc) => {
      const data = doc.data();

      console.log(data);
      return {
        id: doc.id,
        history: {
          result: data.result,
          createdAt: data.createdAt,
          suggestion: data.suggestion,
          id: data.id,
        },
      };
    });

    // Send the response back
    res.json({
      status: "success",
      data: predictions,
    });
  } catch (error) {
    console.error("Error fetching predictions:", error);
    res.status(500).json({
      status: "error",
      message: "Failed to retrieve predictions data",
    });
  }
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
      let tensor = tf.node.decodeImage(imageBuffer, 3);

      const shape = tensor.shape;

      console.log(shape);

      if (shape.length !== 3 || shape[2] !== 3) {
        throw new Error("Image must be in RGB format");
      }

      if (shape[0] > 4000 || shape[1] > 3000) {
        throw new Error("Bad Request!");
      }

      const tensorDecoded = tf.node
        .decodeImage(imageBuffer, 3) // Decode as RGB
        .resizeNearestNeighbor([224, 224]) // Resize to model input size
        .expandDims()
        .div(255.0);

      const prediction = model.predict(tensorDecoded);

      const score = await prediction.data();

      const finalScore = Math.max(...score) * 100;

      console.log(finalScore);

      let label, isBadRequest;
      if (finalScore > 58) {
        label = "Cancer";
        isBadRequest = false;
      } else if (finalScore < 58) {
        label = "Non-cancer";
        isBadRequest = false;
      } else {
        isBadRequest = true;
        label = null;
      }

      const response = {
        status: "success",
        message: "Model is predicted successfully",
        data: {
          id: uuidv4(),
          result: label,
          suggestion:
            label === "Cancer"
              ? "Segera periksa ke dokter!"
              : "Penyakit kanker tidak terdeteksi.",
          createdAt: new Date().toISOString(),
        },
      };

      await db
        .collection("predictions")
        .doc(response.data.id)
        .set(response.data);

      return res.status(201).json(response);
    } catch (error) {
      console.log(error);
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
