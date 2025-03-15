using Microsoft.ML;
using Microsoft.Win32;
using ObjectDetection;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System;
using System.Drawing.Drawing2D;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using ObjectDetection.YoloParser;
using System.Linq;
using ObjectDetection.DataStructures;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Data;
using System.Runtime.InteropServices;

namespace VideoProcessor
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture? capture;
        private bool isProcessing = false;
        private DispatcherTimer? timer;
        private MLContext mlContext = new MLContext();
        private OnnxModelScorer? modelScorer;
        private YoloOutputParser parser;
        private string modelPath;
        private ITransformer? modelTransformer;

        public MainWindow()
        {
            InitializeComponent();
            
            // Initialize model
            var assetsRelativePath = @"assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            modelPath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
            parser = new YoloOutputParser();
            
            try 
            {
                // Check if model file exists
                if (!File.Exists(modelPath))
                {
                    MessageBox.Show($"Model file not found at: {modelPath}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                // Create pipeline for image processing
                var pipeline = mlContext.Transforms.ExtractPixels(
                        outputColumnName: "input_1", 
                        scaleImage: 1f / 255f,
                        interleavePixelColors: true,
                        offsetImage: 0f,
                        inputColumnName: nameof(ImageInputData.Image))
                    .Append(mlContext.Transforms.ApplyOnnxModel(
                        modelFile: modelPath,
                        outputColumnNames: new[] { "grid" },
                        inputColumnNames: new[] { "input_1" }));

                // Create empty data to get input schema
                var emptyData = mlContext.Data.LoadFromEnumerable(new List<ImageInputData>());
                var imagesFolder = Path.Combine(assetsPath, "images");
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

                modelTransformer = pipeline.Fit(imageDataView);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading model: {ex.Message}\nStack trace: {ex.StackTrace}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
        }

        private void OpenButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Video files (*.mp4)|*.mp4|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                StopProcessing();
                SourceVideo.Source = new Uri(openFileDialog.FileName);
                capture = new VideoCapture(openFileDialog.FileName);
            }
        }

        private void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            if (SourceVideo.Source != null)
            {
                SourceVideo.Play();
                StartProcessing();
            }
        }

        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            SourceVideo.Pause();
            StopProcessing();
        }

        private void SourceVideo_MediaOpened(object sender, RoutedEventArgs e)
        {
            timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(33); // ~30 FPS
            timer.Tick += Timer_Tick;
        }

        private void SourceVideo_MediaEnded(object sender, RoutedEventArgs e)
        {
            StopProcessing();
            SourceVideo.Position = TimeSpan.Zero;
        }

        private void StartProcessing()
        {
            if (capture != null && !isProcessing)
            {
                isProcessing = true;
                timer?.Start();
            }
        }

        private void StopProcessing()
        {
            isProcessing = false;
            timer?.Stop();
        }

        private async void Timer_Tick(object? sender, EventArgs e)
        {
            if (capture != null && isProcessing)
            {
                using var frame = new Mat();
                if (capture.Read(frame))
                {
                    var processedFrame = ProcessFrame(frame);
                    await Dispatcher.InvokeAsync(() =>
                    {
                        ProcessedVideo.Source = processedFrame.ToBitmapSource();
                    });
                }
            }
        }

        private Mat ProcessFrame(Mat frame)
        {
            var processedFrame = new Mat();
            var effect = EffectSelector.SelectedIndex;

            switch (effect)
            {
                case 0: // 灰度效果
                    Cv2.CvtColor(frame, processedFrame, ColorConversionCodes.BGR2GRAY);
                    Cv2.CvtColor(processedFrame, processedFrame, ColorConversionCodes.GRAY2BGR);
                    break;
                case 1: // 边缘检测
                    Cv2.Canny(frame, processedFrame, 100, 200);
                    Cv2.CvtColor(processedFrame, processedFrame, ColorConversionCodes.GRAY2BGR);
                    break;
                case 2: // 模糊效果
                    Cv2.GaussianBlur(frame, processedFrame, new OpenCvSharp.Size(15, 15), 0);
                    break;
                case 3: // 物体识别
                    if (modelTransformer != null)
                    {
                        processedFrame = ProcessYoloFrame(frame);
                    }
                    else
                    {
                        frame.CopyTo(processedFrame);
                        Cv2.PutText(processedFrame, "Model not loaded!", new OpenCvSharp.Point(10, 30),
                            HersheyFonts.HersheySimplex, 1.0, new Scalar(0, 0, 255), 2);
                    }
                    break;
                default:
                    frame.CopyTo(processedFrame);
                    break;
            }

            return processedFrame;
        }

        private Mat ProcessYoloFrame(Mat frame)
        {
            var processedFrame = new Mat();
            frame.CopyTo(processedFrame);

            try
            {
                if (modelTransformer == null)
                    throw new InvalidOperationException("Model not loaded");

                // Resize frame to match YOLO input size
                var resizedFrame = new Mat();
                Cv2.Resize(frame, resizedFrame, new OpenCvSharp.Size(416, 416));

                // Convert to byte array
                var imageBytes = new byte[resizedFrame.Total() * resizedFrame.ElemSize()];
                Marshal.Copy(resizedFrame.Data, imageBytes, 0, imageBytes.Length);

                // Create input data
                var imageInputData = new ImageInputData
                {
                    Image = imageBytes
                };

                // Create prediction engine
                var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInputData, ImagePrediction>(modelTransformer);

                // Get prediction
                var prediction = predictionEngine.Predict(imageInputData);

                // Parse outputs
                var boxes = parser.ParseOutputs(prediction.PredictedLabels);
                var filteredBoxes = parser.FilterBoundingBoxes(boxes, 5, 0.5F);

                // Calculate scale factors for bounding box coordinates
                float xScale = (float)frame.Width / 416;
                float yScale = (float)frame.Height / 416;

                // Draw boxes on the frame
                foreach (var box in filteredBoxes)
                {
                    // Get Bounding Box Dimensions and scale them
                    var x = (int)(Math.Max(box.Dimensions.X, 0) * xScale);
                    var y = (int)(Math.Max(box.Dimensions.Y, 0) * yScale);
                    var width = (int)(box.Dimensions.Width * xScale);
                    var height = (int)(box.Dimensions.Height * yScale);

                    // Draw bounding box
                    Cv2.Rectangle(processedFrame, new OpenCvSharp.Point(x, y), 
                        new OpenCvSharp.Point(x + width, y + height), 
                        new Scalar(box.BoxColor.B, box.BoxColor.G, box.BoxColor.R), 2);

                    // Draw label
                    string text = $"{box.Label} ({(box.Confidence * 100):0}%)";
                    var textSize = Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.5, 1, out var baseline);
                    Cv2.Rectangle(processedFrame, 
                        new OpenCvSharp.Point(x, y - textSize.Height - baseline), 
                        new OpenCvSharp.Point(x + textSize.Width, y),
                        new Scalar(box.BoxColor.B, box.BoxColor.G, box.BoxColor.R), -1);
                    Cv2.PutText(processedFrame, text, 
                        new OpenCvSharp.Point(x, y - baseline),
                        HersheyFonts.HersheySimplex, 0.5, new Scalar(255, 255, 255), 1);
                }
            }
            catch (Exception ex)
            {
                Cv2.PutText(processedFrame, $"Error: {ex.Message}", new OpenCvSharp.Point(10, 30),
                    HersheyFonts.HersheySimplex, 1.0, new Scalar(0, 0, 255), 2);
            }

            return processedFrame;
        }

        private void EffectSelector_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            // 效果改变时不需要特殊处理，因为下一帧会自动应用新效果
        }

        protected override void OnClosed(EventArgs e)
        {
            StopProcessing();
            capture?.Dispose();
            base.OnClosed(e);
        }

        string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(MainWindow).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                // Get Bounding Box Dimensions
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                // Resize To Image
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

                // Bounding Box Text
                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("Arial", 12, System.Drawing.FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    System.Drawing.Point atPoint = new System.Drawing.Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                    // Draw text on image 
                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                }
            }

            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }

            image.Save(Path.Combine(outputImageLocation, imageName));
        }

        void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
        {
            Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            }

            Console.WriteLine("");
        }

        // Input data class for ML.NET
        private class ImageInputData
        {
            [ImageType(416, 416)]
            public byte[] Image { get; set; }
        }

        // Output data class for ML.NET
        private class ImagePrediction
        {
            [VectorType(1, 125, 13, 13)]
            public float[] PredictedLabels { get; set; }
        }
    }
} 