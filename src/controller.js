import * as tf from "@tensorflow/tfjs";

export class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }

  addExample(example, label) {
    const y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses)
    );

    if (this.xs == null) {
      //adding first ever example

      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}
