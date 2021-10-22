export interface DetectorConfiguration {
    shiftFactor?: number;
    minSize?: number;
    maxSize?: number;
    scaleFactor?: number;
    iouThreshold?: number;
    memoryBufferEnabled?: boolean;
    memoryBufferSize?: number;
}
