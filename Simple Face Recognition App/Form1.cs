using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.IO;
using System.Threading;
using System.Diagnostics;

namespace Simple_Face_Recognition_App
{
    public partial class Form1 : Form
    {
        #region Variables
        int testid = 0;
        private Capture videoCapture = null;
        private Image<Bgr, Byte> currentFrame = null;
        Mat frame = new Mat();
        private bool facesDetectionEnabled = false;
        CascadeClassifier faceCasacdeClassifier = new CascadeClassifier(@"C:\Users\Pawithra\OneDrive\Desktop\haarcascade_frontalface_default.xml");
        Image<Bgr, Byte> faceResult = null;
        List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, byte>>();
        List<int> PersonsLabes = new List<int>();

        bool EnableSaveImage = false;
        private bool isTrained = false;
        EigenFaceRecognizer recognizer;
        List<string> PersonsNames = new List<string>();
        #endregion
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void btnCapture_Click(object sender, EventArgs e)
        {
            videoCapture = new Capture();
            videoCapture.ImageGrabbed += processFrame;
            videoCapture.Start();
        }

        private void processFrame(object sender, EventArgs e)
        {
            //Step 1: Video Capture
            videoCapture.Retrieve(frame, 0);
            currentFrame = frame.ToImage<Bgr, Byte>().Resize(picCapture.Width, picCapture.Height, Inter.Cubic);

            //Step 2: Face Detection
            if (facesDetectionEnabled)
            {

                //Convert from Bgr to Gray Image
                Mat grayImage = new Mat();
                CvInvoke.CvtColor(currentFrame, grayImage, ColorConversion.Bgr2Gray);
                //Enhance the image to get better result
                CvInvoke.EqualizeHist(grayImage, grayImage);

                Rectangle[] faces = faceCasacdeClassifier.DetectMultiScale(grayImage, 1.1, 3, Size.Empty, Size.Empty);
                //If faces detected
                if (faces.Length > 0)
                {

                    foreach (var face in faces)
                    {
                        //Draw square around each face 
                        // CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                        //Step 3: Add Person 
                        //Assign the face to the picture Box face picDetected
                        Image<Bgr, Byte> resultImage = currentFrame.Convert<Bgr, Byte>();
                        resultImage.ROI = face;
                        picDetected.SizeMode = PictureBoxSizeMode.StretchImage;
                        picDetected.Image = resultImage.Bitmap;

                        if (EnableSaveImage)
                        {
                            //We will create a directory if does not exists!
                            string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                            if (!Directory.Exists(path))
                                Directory.CreateDirectory(path);
                            //we will save 10 images with delay a second for each image 
                            //to avoid hang GUI we will create a new task
                            Task.Factory.StartNew(() =>
                            {
                                for (int i = 0; i < 10; i++)
                                {
                                    //resize the image then saving it
                                    resultImage.Resize(200, 200, Inter.Cubic).Save(path + @"\" + txtPersonName.Text + "_" + DateTime.Now.ToString("dd-mm-yyyy-hh-mm-ss") + ".jpg");
                                    Thread.Sleep(1000);
                                }
                            });

                        }
                        EnableSaveImage = false;

                        if (btnAddPerson.InvokeRequired)
                        {
                            btnAddPerson.Invoke(new ThreadStart(delegate
                            {
                                btnAddPerson.Enabled = true;
                            }));
                        }

                        // Step 5: Recognize the face 
                        if (isTrained)
                        {
                            Image<Gray, Byte> grayFaceResult = resultImage.Convert<Gray, Byte>().Resize(200, 200, Inter.Cubic);
                            CvInvoke.EqualizeHist(grayFaceResult, grayFaceResult);
                            var result = recognizer.Predict(grayFaceResult);
                            pictureBox1.Image = grayFaceResult.Bitmap;
                            pictureBox2.Image = TrainedFaces[result.Label].Bitmap;
                            Debug.WriteLine(result.Label + ". " + result.Distance);
                            //Here results found known faces
                            if (result.Label != -1 && result.Distance < 2000)
                            {
                                CvInvoke.PutText(currentFrame, PersonsNames[result.Label], new Point(face.X - 2, face.Y - 2),
                                    FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Green).MCvScalar, 2);
                            }
                            //here results did not found any know faces
                            else
                            {
                                CvInvoke.PutText(currentFrame, "Unknown", new Point(face.X - 2, face.Y - 2),
                                    FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                            }
                        }
                    }
                }
            }

            //Render the video capture into the Picture Box picCapture
            picCapture.Image = currentFrame.Bitmap;
        }

        private void btnDetectFaces_Click(object sender, EventArgs e)
        {
            facesDetectionEnabled = true;
        }

        private void picDetected_Click(object sender, EventArgs e)
        {

        }

        private void btnAddPerson_Click(object sender, EventArgs e)
        {
            btnAddPerson.Enabled = false;
            EnableSaveImage = true;
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
            TrainImagesFromDir();
        }

        private void txtPersonName_TextChanged(object sender, EventArgs e)
        {

        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            btnSave.Enabled = false;
            btnAddPerson.Enabled = true;
            EnableSaveImage = false;
        }
        //Step 4: train Images .. we will use the saved images from the previous example 
        private bool TrainImagesFromDir()
        {
            int ImagesCount = 0;
            double Threshold = 2000;
            TrainedFaces.Clear();
            PersonsLabes.Clear();
            PersonsNames.Clear();
            try
            {
                string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                string[] files = Directory.GetFiles(path, "*.jpg", SearchOption.AllDirectories);

                foreach (var file in files)
                {
                    Image<Gray, byte> trainedImage = new Image<Gray, byte>(file).Resize(200, 200, Inter.Cubic);
                    CvInvoke.EqualizeHist(trainedImage, trainedImage);
                    TrainedFaces.Add(trainedImage);
                    PersonsLabes.Add(ImagesCount);
                    string name = file.Split('\\').Last().Split('_')[0];
                    PersonsNames.Add(name);
                    ImagesCount++;
                    Debug.WriteLine(ImagesCount + ". " + name);

                }

                if (TrainedFaces.Count() > 0)
                {
                    // recognizer = new EigenFaceRecognizer(ImagesCount,Threshold);
                    recognizer = new EigenFaceRecognizer(ImagesCount, Threshold);
                    recognizer.Train(TrainedFaces.ToArray(), PersonsLabes.ToArray());

                    isTrained = true;
                    //Debug.WriteLine(ImagesCount);
                    //Debug.WriteLine(isTrained);
                    return true;
                }
                else
                {
                    isTrained = false;
                    return false;
                }
            }
            catch (Exception ex)
            {
                return isTrained = false;
                MessageBox.Show("Error in Train Images: " + ex.Message);
            }

        }
    }
}
