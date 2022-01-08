import * as tf from "@tensorflow/tfjs";

const CONTROLS = ["up", "down", "left", "right"];
const CONTROL_CODES = [38, 40, 37, 39];

const trainStatusElement = document.getElementById("train-status");

const statusElement = document.getElementById("status");

export function init() {
  document.getElementById("controller").style.display = "";
  statusElement.style.display = "none";
}

const learningRateElement = document.getElementById("learningRate");
export const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById("batchSizeFraction");
export const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById("epochs");
export const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById("dense-units");
export const getDenseUnits = () => +denseUnitsElement.value;

export const startPacman = () => google.pacman.startGameplay();

export const predictClass = (classId) => {
  google.pacman.keyPressed(CONTROL_CODES[classId]);
  document.body.setAttribute("data-active", CONTROLS[classId]);
};

export function isPredicting() {
  statusElement.style.visibility = "visible";
}
export function donePredicting() {
  statusElement.style.visibility = "hidden";
}
export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

export let addExampleHandler;

export const setExampleHandler = (handler) => (addExampleHandler = handler);

let mouseDown = false;

const totals = [0, 0, 0, 0];

const upButton = document.getElementById("up");
const downButton = document.getElementById("down");
const leftButton = document.getElementById("left");
const rightButton = document.getElementById("right");

const thumbDisplayed = {};

const handler = async (label) => {
  mouseDown = true;

  const className = CONTROLS[label];
  const total = document.getElementById(className + "-total");

  if (mouseDown) {
    addExampleHandler(label);
    document.body.setAttribute("data-active", CONTROLS[label]);
    total.innerText = ++totals[label];
    await tf.nextFrame();
  }
  document.body.removeAttribute("data-active");
};

upButton.addEventListener("mousedown", () => handler(0));
upButton.addEventListener("mouseup", () => (mouseDown = false));

downButton.addEventListener("mousedown", () => handler(1));
downButton.addEventListener("mouseup", () => (mouseDown = false));

leftButton.addEventListener("mousedown", () => handler(2));
leftButton.addEventListener("mouseup", () => (mouseDown = false));

rightButton.addEventListener("mousedown", () => handler(3));
rightButton.addEventListener("mouseup", () => (mouseDown = false));

export const drawThumb = (img, label) => {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + "-thumb");
    draw(img, thumbCanvas);
  }
};

export const draw = (img, canvas) => {
  const [w, h] = [224, 224];
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(w, h);
  /**
   * here img is a tensor image, output of the transferlearning model-1 i.e. the mobilenet model
   */
  const data = img.dataSync();

  for (let i = 0; i < h * w; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i * 3 + 0] * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
};
