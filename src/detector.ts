import {Detection} from "./detection";
import {Classifier} from "./classifier";
import {DetectorConfiguration} from "./detector-configuration";

export class Detector {

    private classifier: Classifier;
    private configuration: DetectorConfiguration;
    private memoryIndex = 0;
    private memoryBuffer: Array<Array<Detection>> = [];

    constructor(classifier: Classifier, configuration?: DetectorConfiguration) {
        this.classifier = classifier;
        this.configuration = Object.assign({
            shiftFactor: 0.1,
            minSize: 100,
            maxSize: 1000,
            scaleFactor: 1.1,
            iouThreshold: 0.2,
            memoryBufferEnabled: false,
            memoryBufferSize: 5
        }, configuration ?? {});
    }

    public setConfiguration(configuration: DetectorConfiguration) {
        Object.assign(this.configuration, configuration);
    }

    public getConfiguration(): DetectorConfiguration {
        return this.configuration;
    }

    public detect(image: ImageData): Array<Detection> {
        let detections = this.findDetections(image);
        if (this.configuration.memoryBufferEnabled) {
            this.updateMemoryBuffer(detections);
            detections = this.getMemoryBufferDetections();
        }
        detections = this.clusterDetections(detections);
        return detections;
    }

    private findDetections(image: ImageData): Array<Detection> {
        const detections: Array<Detection> = [];
        const imagePixels = Detector.getImagePixels(image);
        let scale = this.configuration.minSize;
        while (scale <= this.configuration.maxSize) {
            const step = Math.max(this.configuration.shiftFactor * scale, 1) >> 0;
            const offset = (scale / 2 + 1) >> 0;
            for (let row = offset; row <= image.height - offset; row += step) {
                for (let column = offset; column <= image.width - offset; column += step) {
                    const score = this.classifier.process(row, column, scale, imagePixels, image.width);
                    if (score > 0.0) {
                        detections.push({center: {x: column, y: row}, radius: scale / 2, score});
                    }
                }
            }
            scale = scale * this.configuration.scaleFactor;
        }
        return detections;
    }

    private updateMemoryBuffer(detections: Array<Detection>): void {
        if (this.memoryBuffer.length != this.configuration.memoryBufferSize) {
            this.memoryBuffer = new Array(this.configuration.memoryBufferSize).fill([]);
        }
        this.memoryBuffer[this.memoryIndex] = detections;
        this.memoryIndex = (this.memoryIndex + 1) % this.memoryBuffer.length;
    }

    private getMemoryBufferDetections(): Array<Detection> {
        let memoryDetections: Array<Detection> = [];
        for (let i = 0; i < this.memoryBuffer.length; ++i) {
            memoryDetections = memoryDetections.concat(this.memoryBuffer[i]);
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
                    if (Detector.calculateIOU(detections[i], detections[j]) > this.configuration.iouThreshold) {
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
        for (let y = 0; y < image.height; ++y) {
            for (let x = 0; x < image.width; ++x) {
                imagePixels[y * image.width + x] = (2 * imageData[(y * 4 * image.width + 4 * x)] + 7 * imageData[y * 4 * image.width + 4 * x + 1] + imageData[y * 4 * image.width + 4 * x + 2]) / 10;
            }
        }
        return imagePixels;
    }

    private static calculateIOU(detection1: Detection, detection2: Detection): number {
        const centerY1 = detection1.center.y, centerX1 = detection1.center.x, radius1 = detection1.radius;
        const centerY2 = detection2.center.y, centerX2 = detection2.center.x, radius2 = detection2.radius;
        const overr = Math.max(0, Math.min(centerY1 + radius1 / 2, centerY2 + radius2 / 2) - Math.max(centerY1 - radius1 / 2, centerY2 - radius2 / 2));
        const overc = Math.max(0, Math.min(centerX1 + radius1 / 2, centerX2 + radius2 / 2) - Math.max(centerX1 - radius1 / 2, centerX2 - radius2 / 2));
        return overr * overc / (radius1 * radius1 + radius2 * radius2 - overr * overc);
    }
}
