/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import io.mindmodel.services.common.GraphicsUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.EagerSession;
import org.tensorflow.Tensor;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.nio.nd.LongNdArray;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUInt8;

/**
 *
 * @author Christian Tzolov
 */
public class SemanticSegmentationConfiguration {

	private static final Log logger = LogFactory.getLog(SemanticSegmentationConfiguration.class);

	/**
	 * Blended mask transparency. Value is between 0.0 (0% transparency) and 1.0 (100% transparent).
	 */
	private double maskTransparency = 0.3;

	/**
	 * Generated image format
	 */
	private String imageFormat = "jpg";

	public double getMaskTransparency() {
		return maskTransparency;
	}

	public void setMaskTransparency(double maskTransparency) {
		this.maskTransparency = maskTransparency;
	}

	public String getImageFormat() {
		return imageFormat;
	}

	public void setImageFormat(String imageFormat) {
		this.imageFormat = imageFormat;
	}

	/**
	 * Converts the input image (as byte[]) into input tensor
	 * @return
	 */
	public Function<byte[], Map<String, Tensor<?>>> inputConverter() {
		return image -> {
			BufferedImage scaledImage = SemanticSegmentationUtils.scaledImage(image);
			Tensor<TUInt8> inTensor = SemanticSegmentationUtils.createInputTensor(scaledImage);
			return Collections.singletonMap(SemanticSegmentationUtils.INPUT_TENSOR_NAME, inTensor);
		};
	}

	/**
	 * Converts output named tensors into pixel masks
	 * @return
	 */
	public Function<Map<String, Tensor<?>>, IntNdArray> outputConverter() {
		return resultTensors -> {

			try (Tensor<TInt64> masks =
					resultTensors.get(SemanticSegmentationUtils.OUTPUT_TENSOR_NAME).expect(TInt64.DTYPE)) {

				return SemanticSegmentationUtils.extractOutputData(masks);
			}
		};
	}

	/**
	 * Takes the input image (byte[]) and mask pixels (long matrix) and outputs the same image (byte[]) augmented
	 * with masks overlays.
	 * @return Returns the input image augmented with masks's overlays
	 */
	public BiFunction<byte[], IntNdArray, byte[]> imageAugmenter() {
		return (inputImage, mask) -> {
			try {
				BufferedImage scaledImage = SemanticSegmentationUtils.scaledImage(inputImage);
				BufferedImage maskImage = SemanticSegmentationUtils.createMaskImage(mask, this.getMaskTransparency());
				BufferedImage blend = SemanticSegmentationUtils.blendMask(maskImage, scaledImage);

				return GraphicsUtils.toImageByteArray(blend, this.getImageFormat());
			}
			catch (Exception e) {
				logger.error("Failed to create output message", e);
			}
			return inputImage;
		};
	}

	/**
	 * Converts the pixels (long[][]) into mask image (byte[])
	 * @return Image representing the mask pixels
	 */
	public Function<IntNdArray, byte[]> pixelsToMaskImage() {
		return mask -> {
			try {
				BufferedImage maskImage = SemanticSegmentationUtils.createMaskImage(mask, this.getMaskTransparency());

				return GraphicsUtils.toImageByteArray(maskImage, this.getImageFormat());
			}
			catch (Exception e) {
				logger.error("Failed to create output message", e);
			}
			return new byte[0];
		};
	}

}
