import {Detection} from "./detection";
import {Classifier} from "./classifier";

export class Detector {

    private classifier: Classifier;
    private memoryIndex = 0;
    private memoryBuffer: Array<any> = [];
    private shiftFactor = 0.1;
    private minSize = 100;
    private maxSize = 1000;
    private scaleFactor = 1.1;
    private iouThreshold = 0.2;

    constructor(classifier: Classifier, memoryBufferSize = 1) {
        this.classifier = classifier;
        if (memoryBufferSize > 1) {
            for (let i = 0; i < memoryBufferSize; ++i) {
                this.memoryBuffer.push([]);
            }
        }
    }

    public detect(image: ImageData): Array<Detection> {
        let detections = this.findDetections(image);
        detections = this.findMemoryDetections(detections);
        detections = this.clusterDetections(detections);
        return detections;
    }

    private findDetections(image: ImageData): Array<Detection> {
        const detections: Array<Detection> = [];
        const imagePixels = Detector.getImagePixels(image);
        let scale = this.minSize;
        while (scale <= this.maxSize) {
            const step = Math.max(this.shiftFactor * scale, 1) >> 0;
            const offset = (scale / 2 + 1) >> 0;
            for (let row = offset; row <= image.height - offset; row += step) {
                for (let column = offset; column <= image.width - offset; column += step) {
                    const score = this.classifier.process(row, column, scale, imagePixels, image.width);
                    if (score > 0.0) {
                        detections.push({center: {x: column, y: row}, radius: scale / 2, score});
                    }
                }
            }
            scale = scale * this.scaleFactor;
        }
        return detections;
    }

    private findMemoryDetections(detections: Array<Detection>): Array<Detection> {
        let memoryDetections: Array<Detection> = detections;
        if (this.memoryBuffer.length) {
            this.memoryBuffer[this.memoryIndex] = detections;
            this.memoryIndex = (this.memoryIndex + 1) % this.memoryBuffer.length;
            memoryDetections = [];
            for (let i = 0; i < this.memoryBuffer.length; ++i) {
                memoryDetections = memoryDetections.concat(this.memoryBuffer[i]);
            }
        }
        return memoryDetections;
    }

    private clusterDetections(detections: Array<Detection>): Array<Detection> {
        detections = detections.sort((a, b) => b.score - a.score);
        const assignments = new Array(detections.length).fill(0);
        const clusters: Array<Detection> = [];
        for (let i = 0; i < detections.length; ++i) {
            if (assignments[i] == 0) {
                let centerYSum = 0.0, centerXSum = 0.0, radiusSum = 0.0, scoreSum = 0.0, counter = 0;
                for (let j = i; j < detections.length; ++j) {
                    if (Detector.calculateIOU(detections[i], detections[j]) > this.iouThreshold) {
                        assignments[j] = 1;
                        centerYSum = centerYSum + detections[j].center.y;
                        centerXSum = centerXSum + detections[j].center.x;
                        radiusSum = radiusSum + detections[j].radius;
                        scoreSum = scoreSum + detections[j].score;
                        counter = counter + 1;
                    }
                }
                clusters.push({center: {x: centerXSum / counter, y: centerYSum / counter}, radius: radiusSum / counter, score: scoreSum});
            }
        }
        return clusters;
    }

    private static getImagePixels (image: ImageData): Uint8Array {
        const imageData = image.data;
        const imagePixels = new Uint8Array(image.height * image.width);
        for (let r = 0; r < image.height; ++r) {
            for (let c = 0; c < image.width; ++c) {
                imagePixels[r*image.width + c] = (2 * imageData[(r * 4 * image.width + 4 * c)] + 7 * imageData[r * 4 * image.width + 4 * c + 1] + imageData[r * 4 * image.width + 4 * c + 2]) / 10;
            }
        }
        return imagePixels;
    }

    private static calculateIOU(detection1: Detection, detection2: Detection): number {
        const r1 = detection1.center.y, c1 = detection1.center.x, s1 = detection1.radius;
        const r2 = detection2.center.y, c2 = detection2.center.x, s2 = detection2.radius;
        const overr = Math.max(0, Math.min(r1 + s1 / 2, r2 + s2 / 2) - Math.max(r1 - s1 / 2, r2 - s2 / 2));
        const overc = Math.max(0, Math.min(c1 + s1 / 2, c2 + s2 / 2) - Math.max(c1 - s1 / 2, c2 - s2 / 2));
        return overr * overc / (s1 * s1 + s2 * s2 - overr * overc);
    }
}
