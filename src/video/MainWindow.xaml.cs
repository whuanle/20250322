using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using SkiaSharp;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Threading;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

namespace VideoProcessor
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture? capture;
        private bool isProcessing = false;
        private DispatcherTimer? timer;
        private DateTime lastProcessTime = DateTime.MinValue;
        private const int PROCESS_INTERVAL_MS = 100; // Process every 100ms

        const string assetsRelativePath = @"assets";
        private readonly string assetsPath = GetAbsolutePath(assetsRelativePath);
        private readonly string modelPath;
        private ITransformer? model;
        private DateTime lastYoloProcessTime = DateTime.MinValue;
        private const int YOLO_PROCESS_INTERVAL_MS = 100;

        private readonly Yolo _yolo;

        public MainWindow()
        {
            InitializeComponent();

            // Initialize model path
            modelPath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
            
            // Print model path for debugging
            Console.WriteLine($"Model Path: {modelPath}");
            
            // Check if model file exists
            if (!File.Exists(modelPath))
            {
                // Try to find the model file in the current directory
                var currentDirModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "assets", "Model", "TinyYolo2_model.onnx");
                Console.WriteLine($"Trying alternative path: {currentDirModelPath}");
                
                if (File.Exists(currentDirModelPath))
                {
                    modelPath = currentDirModelPath;
                    Console.WriteLine($"Found model at: {modelPath}");
                }
                else
                {
                    MessageBox.Show($"Model file not found at:\n{modelPath}\nor\n{currentDirModelPath}\n\nPlease make sure the model file exists in the assets/Model directory.",
                        "Model Not Found",
                        MessageBoxButton.OK,
                        MessageBoxImage.Error);
                    return;
                }
            }

            try
            {
                _yolo = new Yolo(new YoloOptions
                {
                    OnnxModel = modelPath,          // Your Yolo model in onnx format
                    ModelType = ModelType.ObjectDetection,      // Set your model type
                    Cuda = false,                               // Use CPU or CUDA for GPU accelerated inference. Default = true
                    GpuId = 0,                                  // Select Gpu by id. Default = 0
                    PrimeGpu = false,                           // Pre-allocate GPU before first inference. Default = false
                });
                Console.WriteLine("YOLO model initialized successfully");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error initializing YOLO model: {ex.Message}\nStack trace: {ex.StackTrace}",
                    "Model Initialization Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
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
                    var processedFrame = await ProcessFrame(frame);
                    await Dispatcher.InvokeAsync(() =>
                    {
                        ProcessedVideo.Source = processedFrame.ToBitmapSource();
                    });
                }
            }
        }

        private async Task<Mat> ProcessFrame(Mat frame)
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
                    if (File.Exists(modelPath))
                    {
                        processedFrame = await ProcessYoloFrame(frame);
                    }
                    else
                    {
                        frame.CopyTo(processedFrame);
                        Cv2.PutText(processedFrame, "Model file not found!", new OpenCvSharp.Point(10, 30),
                            HersheyFonts.HersheySimplex, 1.0, new Scalar(0, 0, 255), 2);
                    }
                    break;
                default:
                    frame.CopyTo(processedFrame);
                    break;
            }

            return processedFrame;
        }

        private async Task<Mat> ProcessYoloFrame(Mat frame)
        {
            var processedFrame = new Mat();
            frame.CopyTo(processedFrame);

            // Convert Mat to Bitmap
            using var bitmap = BitmapConverter.ToBitmap(processedFrame);
            
            // Convert Bitmap to SKImage
            using var ms = new MemoryStream();
            bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
            ms.Position = 0;
            using var skBitmap = SKBitmap.Decode(ms);
            using var image = SKImage.FromBitmap(skBitmap);

            // Run YOLO detection and draw results
            var results = _yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);
            using var resultImage = image.Draw(results);

            // Convert resultImage back to Mat
            using var resultBitmap = SKBitmap.FromImage(resultImage);
            using var data = resultBitmap.Encode(SKEncodedImageFormat.Png, 100);
            
            // Convert to Mat using OpenCV's ImDecode
            var imageData = data.ToArray();
            using var resultMat = Cv2.ImDecode(imageData, ImreadModes.Color);
            resultMat.CopyTo(processedFrame);

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

        static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(MainWindow).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            
            // Print paths for debugging
            Console.WriteLine($"Assembly Location: {_dataRoot.FullName}");
            Console.WriteLine($"Assembly Folder: {assemblyFolderPath}");
            Console.WriteLine($"Relative Path: {relativePath}");
            Console.WriteLine($"Full Path: {fullPath}");
            
            // Check if directory exists
            if (!Directory.Exists(fullPath))
            {
                Console.WriteLine($"Warning: Directory does not exist: {fullPath}");
            }
            else
            {
                Console.WriteLine($"Directory exists: {fullPath}");
                Console.WriteLine("Directory contents:");
                foreach (var file in Directory.GetFiles(fullPath, "*.*", SearchOption.AllDirectories))
                {
                    Console.WriteLine($"  - {file}");
                }
            }

            return fullPath;
        }

        // Input data class for ML.NET
        private class ImageNetData
        {
            [ImageType(416, 416)]
            public byte[] Image { get; set; }
            public string Label { get; set; }
        }

        // Output data class for ML.NET
        private class ImagePrediction
        {
            [VectorType(1, 125, 13, 13)]
            public float[] PredictedLabels { get; set; }
        }

        // Helper class for bitmap conversion
        private static class BitmapConverter
        {
            public static Bitmap ToBitmap(Mat mat)
            {
                // Convert to BGR24 format
                using var bgr = mat.CvtColor(ColorConversionCodes.BGR2RGB);
                // Create bitmap with the same dimensions
                var bitmap = new Bitmap(mat.Width, mat.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                // Lock bits to get direct access to pixel data
                var bitmapData = bitmap.LockBits(
                    new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height),
                    System.Drawing.Imaging.ImageLockMode.WriteOnly,
                    bitmap.PixelFormat);

                try
                {
                    // Copy pixel data
                    var length = Math.Min((int)bgr.Total() * bgr.ElemSize(), Math.Abs(bitmapData.Stride) * bitmap.Height);
                    var data = new byte[length];
                    Marshal.Copy(bgr.Data, data, 0, length);
                    Marshal.Copy(data, 0, bitmapData.Scan0, length);
                    return bitmap;
                }
                finally
                {
                    bitmap.UnlockBits(bitmapData);
                }
            }
        }
    }
}