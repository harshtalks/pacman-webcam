import * as tf from "@tensorflow/tfjs";
import * as tfd from "@tensorflow/tfjs-data";

import { ControllerDataset } from "./controller";

import * as ui from "./ui";

const NUM_CLASSES = 4;
let webcam;
let model;
let truncatedMobileNetModel;
const LAYER = "conv_pw_13_relu";

const controllerDataset = new ControllerDataset(NUM_CLASSES);

console.log(controllerDataset);

const loadTruncatedMonileNetModel = async () => {
  const url =
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json";

  const mobileNet = await tf.loadLayersModel(url);

  console.log("loaded mobilenet model is: ", mobileNet);

  const layer = mobileNet.getLayer(LAYER);

  return tf.model({
    inputs: mobileNet.inputs,
    outputs: layer.output
  });
};

ui.setExampleHandler(async (label) => {
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNetModel.predict(img), label);

  ui.drawThumb(img, label);
  img.dispose();
});

const getImage = async () => {
  const img = await webcam.capture();
  const processedImg = tf.tidy(() =>
    img.expandDims(0).toFloat().div(127).sub(1)
  );
  img.dispose();
  return processedImg;
};

const train = async () => {
  if (controllerDataset.xs == null) {
    throw new Error("Add some examples before training");
  }

  model = tf.sequential({
    layers: [
      tf.layers.flatten({
        inputShape: truncatedMobileNetModel.outputs[0].shape.slice(1)
      }),
      // Layer 1.
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: "relu",
        kernelInitializer: "varianceScaling",
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: "varianceScaling",
        useBias: false,
        activation: "softmax"
      })
    ]
  });

  const optimizer = tf.train.adam(ui.getLearningRate());

  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });

  const batchSize = Math.floor(
    controllerDataset.xs.shape[0] * ui.getBatchSizeFraction()
  );
  if (!(batchSize > 0)) {
    throw new Error(
      `Batch size is 0 or NaN. Please choose a non-zero fraction.`
    );
  }

  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus("Loss: " + logs.loss.toFixed(5));
      }
    }
  });
};

let isPredicting = false;

const predict = async () => {
  ui.isPredicting();
  while (isPredicting) {
    const img = await getImage();

    const embeddings = truncatedMobileNetModel.predict(img);

    const predictions = model.predict(embeddings);

    const predictedClass = predictions.as1D().argMax();
    const classId = await predictedClass.dataSync()[0];
    img.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }

  ui.donePredicting();
};

document.getElementById("train").addEventListener("click", async () => {
  ui.trainStatus("Training...");
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById("predict").addEventListener("click", () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

const init = async () => {
  try {
    webcam = await tfd.webcam(document.getElementById("webcam"));
  } catch (e) {
    console.error(e);
    document.getElementById("no-webcam").style.display = "block";
  }
  truncatedMobileNetModel = await loadTruncatedMonileNetModel();
  ui.init();
};

init();
