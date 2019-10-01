package io.mindmodel.services.semantic.segmentation;

import static java.awt.image.BufferedImage.TYPE_3BYTE_BGR;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.mindmodel.services.common.GraphicsUtils;
import io.mindmodel.services.common.TensorFlowService;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.Map;
import javax.imageio.ImageIO;
import org.springframework.core.io.DefaultResourceLoader;
import org.tensorflow.EagerSession;
import org.tensorflow.Tensor;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUInt8;

/**
 *
 * Semantic image segmentation - the task of assigning a semantic label, such as “road”, “sky”, “person”, “dog”, to
 * every pixel in an image.
 *
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 * https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
 * http://presentations.cocodataset.org/Places17-GMRI.pdf
 *
 * http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
 * https://www.cityscapes-dataset.com/dataset-overview/#class-definitions
 * http://groups.csail.mit.edu/vision/datasets/ADE20K/
 *
 * https://github.com/mapillary/inplace_abn
 *
 * @author Christian Tzolov
 */
public class SemanticSegmentationUtils {

	public static final String INPUT_TENSOR_NAME = "ImageTensor:0";
	public static final String OUTPUT_TENSOR_NAME = "SemanticPredictions:0";

	private static final int BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	private static final int REQUIRED_INPUT_IMAGE_SIZE = 513;

	public static BufferedImage scaledImage(String imagePath) {
		try {
			return scaledImage(ImageIO.read(new DefaultResourceLoader().getResource(imagePath).getInputStream()));
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load Image from: " + imagePath, e);
		}
	}

	public static BufferedImage scaledImage(byte[] image) {
		try {
			return scaledImage(ImageIO.read(new ByteArrayInputStream(image)));
		}
		catch (IOException e) {
			throw new IllegalStateException("Failed to load Image from byte array", e);
		}
	}

	public static BufferedImage scaledImage(BufferedImage image) {
		double scaleRatio = 1.0 * REQUIRED_INPUT_IMAGE_SIZE / Math.max(image.getWidth(), image.getHeight());
		return scale(image, scaleRatio);
	}

	private static BufferedImage scale(BufferedImage originalImage, double scale) {
		int newWidth = (int) (originalImage.getWidth() * scale);
		int newHeight = (int) (originalImage.getHeight() * scale);

		Image tmpImage = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT);
		//BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_INT_BGR);
		BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, TYPE_3BYTE_BGR);
		//BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, originalImage.getType());

		Graphics2D g2d = resizedImage.createGraphics();
		g2d.drawImage(tmpImage, 0, 0, null);
		g2d.dispose();

		return resizedImage;
	}

	public static BufferedImage blendMask(BufferedImage mask, BufferedImage background) {
		GraphicsUtils.overlayImages(background, mask, 0, 0);
		return background;
	}

	public static Tensor<TUInt8> createInputTensor(Ops tf, BufferedImage scaledImage) {
		if (scaledImage.getType() != TYPE_3BYTE_BGR) {
			throw new IllegalArgumentException(
					String.format("Expected 3-byte BGR encoding in BufferedImage, found %d", scaledImage.getType()));
		}

		// Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		Shape shape = Shape.make(BATCH_SIZE, scaledImage.getHeight(), scaledImage.getWidth(), CHANNELS);

		try (Tensor<TUInt8> tensor = TUInt8.tensorOfShape(shape)) {
			tensor.data().write(toBytes(scaledImage));

			// ImageIO.read produces BGR-encoded images, while the model expects RGB.
			return tf.reverse(tf.constant(tensor), tf.constant(new int[]{3})).asOutput().tensor();
		}
	}

	private static byte[] toBytes(BufferedImage bufferedImage) {
		return ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	}

	public static IntNdArray extractOutputData(Ops tf, Tensor<TInt64> outputTensor) {
		return tf.dtypes.cast(
				tf.linalg.transpose(
						tf.squeeze(tf.constant(outputTensor)), // squeeze tensor to remove batch dimension of size 1
						tf.constant(new int[] {1, 0})
				),
				TInt32.DTYPE
		).asOutput().tensor().data(); // FIXME! that leaves potentially one tensor open for a long time...
	}

	public static BufferedImage createMaskImage(IntNdArray mask, double transparency) {
		return createMaskImage(mask, (int) mask.shape().size(1), (int) mask.shape().size(0), transparency);
	}

	public static BufferedImage createMaskImage(IntNdArray maskPixels, int width, int height, double transparency) {

		int maskWidth = (int) maskPixels.shape().size(0);
		int maskHeight = (int) maskPixels.shape().size(1);

		int[] maskArray = new int[maskWidth * maskHeight];
		int k = 0;
		for (int i = 0; i < maskHeight; i++) {
			for (int j = 0; j < maskWidth; j++) {
				Color c = (maskPixels.get(j, i) == 0) ? Color.BLACK : GraphicsUtils.getClassColor(maskPixels.get(j, i));
				int t = (int) (255 * (1 - transparency));
				maskArray[k++] = new Color(c.getRed(), c.getGreen(), c.getBlue(), t).getRGB();
			}
		}

		// Turn the pixel array into image;
		BufferedImage maskImage = new BufferedImage(maskWidth, maskHeight, BufferedImage.TYPE_INT_ARGB);
		maskImage.setRGB(0, 0, maskWidth, maskHeight, maskArray, 0, maskWidth);

		// Stretch the image to fit the target box width and height!
		return GraphicsUtils.toBufferedImage(maskImage.getScaledInstance(width, height, Image.SCALE_SMOOTH));
	}

	public String serializeToJson(int[][] pixels) {
		String masksBase64 = Base64.getEncoder().encodeToString(toBytes(pixels));
		return String.format("{ \"columns\":%d, \"rows\":%d, \"masks\":\"%s\"}", pixels.length, pixels[0].length, masksBase64);
	}

	public int[][] deserializeToMasks(String json) throws IOException {
		Map<String, Object> map = new ObjectMapper().readValue(json, Map.class);
		int cols = (int) map.get("columns");
		int rows = (int) map.get("rows");
		String masksBase64 = (String) map.get("masks");
		byte[] masks = Base64.getDecoder().decode(masksBase64);
		return toInts(masks, cols, rows);
	}

	private byte[] toBytes(int[][] pixels) {
		byte[] b = new byte[pixels.length * pixels[0].length * 4];
		int bi = 0;
		for (int i = 0; i < pixels.length; i++) {
			for (int j = 0; j < pixels[0].length; j++) {
				b[bi + 0] = (byte) (i >> 24);
				b[bi + 1] = (byte) (i >> 16);
				b[bi + 2] = (byte) (i >> 8);
				b[bi + 3] = (byte) (i /*>> 0*/);
				bi = bi + 4;
			}
		}
		return b;
	}

	private int[][] toInts(byte[] b, int ic, int jc) {
		int[][] intResult = new int[ic][jc];
		int bi = 0;
		for (int i = 0; i < ic; i++) {
			for (int j = 0; j < jc; j++) {
				intResult[i][j] = (b[bi] << 24) +
						(b[bi + 1] << 16) +
						(b[bi + 2] << 8) +
						b[bi + 3];
				bi = bi + 4;
			}
		}
		return intResult;
	}

	public static void main(String[] args) throws IOException {

		// PASCAL VOC 2012
		//String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/VikiMaxiAdi.jpg";

		// CITYSCAPE
		//String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/amsterdam-cityscape1.jpg";
		//String imagePath = "classpath:/images/amsterdam-channel.jpg";
		//String imagePath = "classpath:/images/landsmeer.png";

		// ADE20K
		//String tensorflowModelLocation = "file:/Users/ctzolov/Downloads/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb";
		String tensorflowModelLocation = "http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz#frozen_inference_graph.pb";
		//String imagePath = "classpath:/images/interior.jpg";
		String imagePath = "classpath:/images/VikiMaxiAdi.jpg";

		BufferedImage inputImage = ImageIO.read(new DefaultResourceLoader().getResource(imagePath).getInputStream());

		TensorFlowService tf = new TensorFlowService(new DefaultResourceLoader().getResource(tensorflowModelLocation), Arrays.asList(OUTPUT_TENSOR_NAME));
		Ops ops = Ops.create(EagerSession.getDefault());

		BufferedImage scaledImage = scaledImage(inputImage);

		Tensor<TUInt8> inTensor = createInputTensor(ops, scaledImage);

		Map<String, Tensor<?>> output = tf.apply(Collections.singletonMap(INPUT_TENSOR_NAME, inTensor));

		IntNdArray maskPixels = extractOutputData(ops, output.get(OUTPUT_TENSOR_NAME).expect(TInt64.DTYPE));

		BufferedImage maskImage = createMaskImage(maskPixels, scaledImage.getWidth(), scaledImage.getHeight(), 0.35);

		BufferedImage blended = blendMask(maskImage, scaledImage);

		ImageIO.write(maskImage, "png", new File("./semantic-segmentation/target/java2Dmask.jpg"));
		ImageIO.write(blended, "png", new File("./semantic-segmentation/target/java2Dblended.jpg"));
	}
}
