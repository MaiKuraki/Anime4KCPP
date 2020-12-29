#define DLL

#include "OpenCLACNet.hpp"

#define CLEAN_KERNEL_AND_THROW_ERROR(err, errCode) \
{\
clReleaseMemObject(imageBufferOrg); \
clReleaseMemObject(imageBufferTmp1); \
clReleaseMemObject(imageBufferTmp2); \
clReleaseMemObject(imageBufferDst); \
clReleaseKernel(kernelConv1To8L1); \
clReleaseKernel(kernelConv8To8L2); \
clReleaseKernel(kernelConv8To8L3); \
clReleaseKernel(kernelConv8To8L4); \
clReleaseKernel(kernelConv8To8L5); \
clReleaseKernel(kernelConv8To8L6); \
clReleaseKernel(kernelConv8To8L7); \
clReleaseKernel(kernelConv8To8L8); \
clReleaseKernel(kernelConv8To8L9); \
clReleaseKernel(kernelConvTranspose8To1L10); \
throw ACException<ExceptionType::GPU, true>(err, errCode); \
}


Anime4KCPP::OpenCL::ACNet::ACNet(const Parameters& parameters) :
    AC(parameters) 
{
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            currACNetypeIndex = HDNL1;
            break;
        case 2:
            currACNetypeIndex = HDNL2;
            break;
        case 3:
            currACNetypeIndex = HDNL3;
            break;
        default:
            currACNetypeIndex = HDNL1;
            break;
        }
    }
    else
    {
        currACNetypeIndex = HDNL0;
    }
}

void Anime4KCPP::OpenCL::ACNet::setArguments(const Parameters& parameters)
{
    AC::setArguments(parameters);
    if (param.HDN)
    {
        switch (param.HDNLevel)
        {
        case 1:
            currACNetypeIndex = HDNL1;
            break;
        case 2:
            currACNetypeIndex = HDNL2;
            break;
        case 3:
            currACNetypeIndex = HDNL3;
            break;
        default:
            currACNetypeIndex = HDNL1;
            break;
        }
    }
    else
    {
        currACNetypeIndex = HDNL0;
    }
}

void Anime4KCPP::OpenCL::ACNet::initGPU(unsigned int platformID, unsigned int deviceID, const CNNType type, const int OpenCLQueueNum, const bool OpenCLParallelIO)
{
    if (!isInitialized)
    {
        pID = platformID;
        dID = deviceID;
        commandQueueNum = OpenCLQueueNum >= 1 ? OpenCLQueueNum : 1;
        parallelIO = OpenCLParallelIO;
        initOpenCL(type);
        isInitialized = true;
    }
}

void Anime4KCPP::OpenCL::ACNet::releaseGPU() noexcept
{
    if (isInitialized)
    {
        releaseOpenCL();
        context = nullptr;
        std::fill(commandQueueList.begin(), commandQueueList.end(), nullptr);
        commandQueueIO = nullptr;
        for (int i = HDNL0; i < TotalTypeCount; i++)
            program[i] = nullptr;
        device = nullptr;
        isInitialized = false;
    }
}

bool Anime4KCPP::OpenCL::ACNet::isInitializedGPU()
{
    return isInitialized;
}

std::string Anime4KCPP::OpenCL::ACNet::getInfo()
{
    std::ostringstream oss;
    oss << AC::getInfo()
        << "----------------------------------------------" << std::endl
        << "OpenCL Platform ID:" << pID << std::endl
        << "OpenCL Device ID:" << dID << std::endl
        << "Zoom Factor: " << param.zoomFactor << std::endl
        << "HDN Mode: " << std::boolalpha << param.HDN << std::endl
        << "HDN Level: " << (param.HDN ? param.HDNLevel : 0) << std::endl
        << "Number of OpenCL Command Queues:" << commandQueueNum << std::endl
        << "OpenCL Parallel IO Command Queues:" << std::boolalpha << parallelIO << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

std::string Anime4KCPP::OpenCL::ACNet::getFiltersInfo()
{
    std::ostringstream oss;
    oss << AC::getFiltersInfo()
        << "----------------------------------------------" << std::endl
        << "Filter not supported" << std::endl
        << "----------------------------------------------" << std::endl;
    return oss.str();
}

void Anime4KCPP::OpenCL::ACNet::processYUVImageB()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpY = orgY;
        dstU = orgU;
        dstV = orgV;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_8UC1);
            if(parallelIO)
                runKernelPB(tmpY, dstY);
            else
                runKernelB(tmpY, dstY);

            cv::resize(dstU, dstU, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            cv::resize(dstV, dstV, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            tmpY = dstY;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstY, dstY, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstU, dstU, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstV, dstV, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_8UC1);
        if (parallelIO)
            runKernelPB(orgY, dstY);
        else
            runKernelB(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBImageB()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            if (parallelIO)
                runKernelPB(tmpImg, dstImg);
            else
                runKernelB(tmpImg, dstImg);
            cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            tmpImg = dstImg;
        }

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
        if (tmpZfUp - tmpZf > 0.00001)
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_8UC1);
        if (parallelIO)
            runKernelPB(orgImg, dstImg);
        else
            runKernelB(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::OpenCL::ACNet::processGrayscaleB()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_8UC1);
            if (parallelIO)
                runKernelPB(tmpImg, dstImg);
            else
                runKernelB(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstImg, dstImg, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_8UC1);
        if (parallelIO)
            runKernelPB(orgImg, dstImg);
        else
            runKernelB(orgImg, dstImg);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBVideoB()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        videoIO->init(
            [this, tmpZfUp, tmpZf]()
            {
                Utils::Frame frame = videoIO->read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                cv::Mat tmpFrame = orgFrame;
                cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2YUV);

                std::vector<cv::Mat> yuv(3);
                cv::split(tmpFrame, yuv);
                tmpFrame = yuv[Y];

                for (int i = 0; i < tmpZfUp; i++)
                {
                    dstFrame.create(tmpFrame.rows * 2, tmpFrame.cols * 2, CV_8UC1);
                    if (parallelIO)
                        runKernelPB(tmpFrame, dstFrame);
                    else
                        runKernelB(tmpFrame, dstFrame);

                    cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
                    cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
                    tmpFrame = dstFrame;
                }

                cv::merge(std::vector<cv::Mat>{ dstFrame, yuv[U], yuv[V] }, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);
                if (tmpZfUp - tmpZf > 0.00001)
                {
                    cv::resize(dstFrame, dstFrame, cv::Size(W, H), 0, 0, cv::INTER_AREA);
                }

                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
    else
    {
        videoIO->init(
            [this]()
            {
                Utils::Frame frame = videoIO->read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame;

                if (param.zoomFactor > 2.0)
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
                else if (param.zoomFactor < 2.0)
                    cv::resize(orgFrame, orgFrame, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

                cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2YUV);

                std::vector<cv::Mat> yuv(3);
                cv::split(orgFrame, yuv);
                orgFrame = yuv[Y];

                dstFrame.create(orgFrame.rows * 2, orgFrame.cols * 2, CV_8UC1);
                if (parallelIO)
                    runKernelPB(orgFrame, dstFrame);
                else
                    runKernelB(orgFrame, dstFrame);

                cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
                cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

                cv::merge(std::vector<cv::Mat>{ dstFrame, yuv[U], yuv[V] }, dstFrame);
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_YUV2BGR);

                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
}

void Anime4KCPP::OpenCL::ACNet::processYUVImageW()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpY = orgY;
        dstU = orgU;
        dstV = orgV;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_16UC1);
            if (parallelIO)
                runKernelPW(tmpY, dstY);
            else
                runKernelW(tmpY, dstY);

            cv::resize(dstU, dstU, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            cv::resize(dstV, dstV, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            tmpY = dstY;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstY, dstY, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstU, dstU, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstV, dstV, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_16UC1);
        if (parallelIO)
            runKernelPW(orgY, dstY);
        else
            runKernelW(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBImageW()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_16UC1);
            if (parallelIO)
                runKernelPW(tmpImg, dstImg);
            else
                runKernelW(tmpImg, dstImg);
            cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            tmpImg = dstImg;
        }

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
        if (tmpZfUp - tmpZf > 0.00001)
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_16UC1);
        if (parallelIO)
            runKernelPW(orgImg, dstImg);
        else
            runKernelW(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::OpenCL::ACNet::processGrayscaleW()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_16UC1);
            if (parallelIO)
                runKernelPW(tmpImg, dstImg);
            else
                runKernelW(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstImg, dstImg, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_16UC1);
        if (parallelIO)
            runKernelPW(orgImg, dstImg);
        else
            runKernelW(orgImg, dstImg);
    }
}

void Anime4KCPP::OpenCL::ACNet::processYUVImageF()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpY = orgY;
        dstU = orgU;
        dstV = orgV;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstY.create(tmpY.rows * 2, tmpY.cols * 2, CV_32FC1);
            if (parallelIO)
                runKernelPF(tmpY, dstY);
            else
                runKernelF(tmpY, dstY);

            cv::resize(dstU, dstU, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            cv::resize(dstV, dstV, cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            tmpY = dstY;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstY, dstY, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstU, dstU, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
            cv::resize(dstV, dstV, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgY, orgY, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstY.create(orgY.rows * 2, orgY.cols * 2, CV_32FC1);
        if (parallelIO)
            runKernelPF(orgY, dstY);
        else
            runKernelF(orgY, dstY);

        cv::resize(orgU, dstU, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        cv::resize(orgV, dstV, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
    }
}

void Anime4KCPP::OpenCL::ACNet::processRGBImageF()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        cv::cvtColor(tmpImg, tmpImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(tmpImg, yuv);
        tmpImg = yuv[Y];

        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_32FC1);
            if (parallelIO)
                runKernelPF(tmpImg, dstImg);
            else
                runKernelF(tmpImg, dstImg);

            cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
            tmpImg = dstImg;
        }

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
        if (tmpZfUp - tmpZf > 0.00001)
        {
            cv::resize(dstImg, dstImg, cv::Size(W, H), 0, 0, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        cv::cvtColor(orgImg, orgImg, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> yuv(3);
        cv::split(orgImg, yuv);
        orgImg = yuv[Y];

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_32FC1);
        if (parallelIO)
            runKernelPF(orgImg, dstImg);
        else
            runKernelF(orgImg, dstImg);

        cv::resize(yuv[U], yuv[U], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);
        cv::resize(yuv[V], yuv[V], cv::Size(0, 0), 2.0, 2.0, cv::INTER_CUBIC);

        cv::merge(std::vector<cv::Mat>{ dstImg, yuv[U], yuv[V] }, dstImg);
        cv::cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);
    }
}

void Anime4KCPP::OpenCL::ACNet::processGrayscaleF()
{
    if (!param.fastMode)
    {
        double tmpZf = std::log2(param.zoomFactor);
        if (tmpZf < 0.0001)
            tmpZf = 1.0 - 0.0002;
        int tmpZfUp = std::ceil(tmpZf);

        cv::Mat tmpImg = orgImg;
        for (int i = 0; i < tmpZfUp; i++)
        {
            dstImg.create(tmpImg.rows * 2, tmpImg.cols * 2, CV_32FC1);
            if (parallelIO)
                runKernelPF(tmpImg, dstImg);
            else
                runKernelF(tmpImg, dstImg);
            tmpImg = dstImg;
        }
        if (tmpZfUp - tmpZf > 0.00001)
        {
            double currZf = param.zoomFactor / exp2(tmpZfUp);
            cv::resize(dstImg, dstImg, cv::Size(0, 0), currZf, currZf, cv::INTER_AREA);
        }
    }
    else
    {
        if (param.zoomFactor > 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_CUBIC);
        else if (param.zoomFactor < 2.0)
            cv::resize(orgImg, orgImg, cv::Size(0, 0), param.zoomFactor / 2.0, param.zoomFactor / 2.0, cv::INTER_AREA);

        dstImg.create(orgImg.rows * 2, orgImg.cols * 2, CV_32FC1);
        if (parallelIO)
            runKernelPF(orgImg, dstImg);
        else
            runKernelF(orgImg, dstImg);
    }
}

void Anime4KCPP::OpenCL::ACNet::runKernelB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);
}

void Anime4KCPP::OpenCL::ACNet::runKernelW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT16;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);
}

void Anime4KCPP::OpenCL::ACNet::runKernelF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueue, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadImage(commandQueue, imageBufferDst, CL_TRUE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 0, nullptr, nullptr);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPB(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
        //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
        //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
        //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
        //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
        //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
        //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
        //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
        //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
        //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueueIO, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 1, &writeFinishedEvent, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPW(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_UNORM_INT16;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
    //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
    //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
    //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
    //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
    //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
    //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
    //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
    //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
    //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueueIO, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 1, &writeFinishedEvent, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::ACNet::runKernelPF(const cv::Mat& orgImg, cv::Mat& dstImg)
{
    cl_int err = CL_SUCCESS;

    cl_event writeFinishedEvent = nullptr;
    cl_event readReadyEvent = nullptr;
    cl_event readFinishedEvent = nullptr;

    cl_image_format format{};
    cl_image_format tmpFormat{};

    cl_image_desc dstDesc{};
    cl_image_desc tmpDesc{};
    cl_image_desc orgDesc{};

    constexpr size_t orgin[3] = { 0,0,0 };
    const size_t orgRegion[3] = { static_cast<const size_t>(orgImg.cols),static_cast<const size_t>(orgImg.rows),1 };
    const size_t dstRegion[3] = { static_cast<const size_t>(dstImg.cols),static_cast<const size_t>(dstImg.rows),1 };

    const size_t orgSize[2] =
    {
        (((static_cast<const size_t>(orgImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(orgImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };
    const size_t dstSize[2] =
    {
        (((static_cast<const size_t>(dstImg.cols) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog,
        (((static_cast<const size_t>(dstImg.rows) - 1) >> workGroupSizeLog) + 1) << workGroupSizeLog
    };

    //init frame
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_R;

    tmpFormat.image_channel_data_type = CL_FLOAT;
    tmpFormat.image_channel_order = CL_RGBA;

    orgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    orgDesc.image_height = orgImg.rows;
    orgDesc.image_width = orgImg.cols;
    orgDesc.buffer = nullptr;

    tmpDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    tmpDesc.image_height = orgImg.rows;
    tmpDesc.image_width = orgImg.cols;
    tmpDesc.image_array_size = 2;
    tmpDesc.buffer = nullptr;

    dstDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    dstDesc.image_height = dstImg.rows;
    dstDesc.image_width = dstImg.cols;
    dstDesc.buffer = nullptr;

    cl_command_queue commandQueue = commandQueueList[commandQueueCount++];
    if (commandQueueCount >= commandQueueNum)
        commandQueueCount = 0;

    cl_kernel kernelConv1To8L1 = clCreateKernel(program[currACNetypeIndex], "conv1To8", &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L1", err);
    }
    cl_kernel kernelConv8To8L2 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L2", err);
    }
    cl_kernel kernelConv8To8L3 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L3", err);
    }
    cl_kernel kernelConv8To8L4 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L4", err);
    }
    cl_kernel kernelConv8To8L5 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L5", err);
    }
    cl_kernel kernelConv8To8L6 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L6", err);
    }
    cl_kernel kernelConv8To8L7 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L7", err);
    }
    cl_kernel kernelConv8To8L8 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L8", err);
    }
    cl_kernel kernelConv8To8L9 = clCreateKernel(program[currACNetypeIndex], "conv8To8", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L9", err);
    }
    cl_kernel kernelConvTranspose8To1L10 = clCreateKernel(program[currACNetypeIndex], "convTranspose8To1", &err);
    if (err != CL_SUCCESS)
    {
        clReleaseKernel(kernelConv1To8L1);
        clReleaseKernel(kernelConv8To8L2);
        clReleaseKernel(kernelConv8To8L3);
        clReleaseKernel(kernelConv8To8L4);
        clReleaseKernel(kernelConv8To8L5);
        clReleaseKernel(kernelConv8To8L6);
        clReleaseKernel(kernelConv8To8L7);
        clReleaseKernel(kernelConv8To8L8);
        clReleaseKernel(kernelConv8To8L9);
        throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel L10", err);
    }


    cl_mem imageBufferOrg = clCreateImage(context, CL_MEM_READ_ONLY, &format, &orgDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw ACException<ExceptionType::GPU, true>("Request imageBufferOrg error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp1 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp1 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferTmp2 = clCreateImage(context, CL_MEM_READ_WRITE, &tmpFormat, &tmpDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferTmp2 error, video memory may be insufficient.", err);
    }

    cl_mem imageBufferDst = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &dstDesc, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        clReleaseMemObject(imageBufferOrg);
        clReleaseMemObject(imageBufferTmp1);
        clReleaseMemObject(imageBufferTmp2);
        throw ACException<ExceptionType::GPU, true>("Request imageBufferDst error, video memory may be insufficient.", err);
    }

    //L1
    err = clSetKernelArg(kernelConv1To8L1, 0, sizeof(cl_mem), &imageBufferOrg);
    err |= clSetKernelArg(kernelConv1To8L1, 1, sizeof(cl_mem), &imageBufferTmp1);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L1 clSetKernelArg error", err)
        //L2
    err = clSetKernelArg(kernelConv8To8L2, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L2, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L2, 2, sizeof(cl_int), &L2);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L2 clSetKernelArg error", err)
        //L3
    err = clSetKernelArg(kernelConv8To8L3, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L3, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L3, 2, sizeof(cl_int), &L3);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L3 clSetKernelArg error", err)
        //L4
    err = clSetKernelArg(kernelConv8To8L4, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L4, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L4, 2, sizeof(cl_int), &L4);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L4 clSetKernelArg error", err)
        //L5
    err = clSetKernelArg(kernelConv8To8L5, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L5, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L5, 2, sizeof(cl_int), &L5);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L5 clSetKernelArg error", err)
        //L6
    err = clSetKernelArg(kernelConv8To8L6, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L6, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L6, 2, sizeof(cl_int), &L6);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L6 clSetKernelArg error", err)
        //L7
    err = clSetKernelArg(kernelConv8To8L7, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L7, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L7, 2, sizeof(cl_int), &L7);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L7 clSetKernelArg error", err)
        //L8
    err = clSetKernelArg(kernelConv8To8L8, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L8, 1, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L8, 2, sizeof(cl_int), &L8);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L8 clSetKernelArg error", err)
        //L9
    err = clSetKernelArg(kernelConv8To8L9, 0, sizeof(cl_mem), &imageBufferTmp2);
    err |= clSetKernelArg(kernelConv8To8L9, 1, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConv8To8L9, 2, sizeof(cl_int), &L9);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L9 clSetKernelArg error", err)
        //L10
    err = clSetKernelArg(kernelConvTranspose8To1L10, 0, sizeof(cl_mem), &imageBufferTmp1);
    err |= clSetKernelArg(kernelConvTranspose8To1L10, 1, sizeof(cl_mem), &imageBufferDst);
    if (err != CL_SUCCESS)
        CLEAN_KERNEL_AND_THROW_ERROR("L10 clSetKernelArg error", err)

    clEnqueueWriteImage(commandQueueIO, imageBufferOrg, CL_FALSE, orgin, orgRegion, orgImg.step, 0, orgImg.data, 0, nullptr, &writeFinishedEvent);
    clEnqueueNDRangeKernel(commandQueue, kernelConv1To8L1, 2, nullptr, orgSize, nullptr, 1, &writeFinishedEvent, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L2, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L3, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L4, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L5, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L6, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L7, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L8, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConv8To8L9, 2, nullptr, orgSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(commandQueue, kernelConvTranspose8To1L10, 2, nullptr, dstSize, nullptr, 0, nullptr, &readReadyEvent);
    clEnqueueReadImage(commandQueueIO, imageBufferDst, CL_FALSE, orgin, dstRegion, dstImg.step, 0, dstImg.data, 1, &readReadyEvent, &readFinishedEvent);

    clWaitForEvents(1, &readFinishedEvent);

    //clean
    clReleaseMemObject(imageBufferOrg);
    clReleaseMemObject(imageBufferTmp1);
    clReleaseMemObject(imageBufferTmp2);
    clReleaseMemObject(imageBufferDst);

    clReleaseKernel(kernelConv1To8L1);
    clReleaseKernel(kernelConv8To8L2);
    clReleaseKernel(kernelConv8To8L3);
    clReleaseKernel(kernelConv8To8L4);
    clReleaseKernel(kernelConv8To8L5);
    clReleaseKernel(kernelConv8To8L6);
    clReleaseKernel(kernelConv8To8L7);
    clReleaseKernel(kernelConv8To8L8);
    clReleaseKernel(kernelConv8To8L9);
    clReleaseKernel(kernelConvTranspose8To1L10);

    clReleaseEvent(writeFinishedEvent);
    clReleaseEvent(readReadyEvent);
    clReleaseEvent(readFinishedEvent);
}

void Anime4KCPP::OpenCL::ACNet::initOpenCL(const CNNType type)
{
    cl_int err = CL_SUCCESS;
    cl_uint platforms = 0;
    cl_uint devices = 0;
    cl_platform_id currentplatform = nullptr;

    //init platform
    err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || !platforms)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL platform", err);
    }

    cl_platform_id* tmpPlatform = new cl_platform_id[platforms];
    err = clGetPlatformIDs(platforms, tmpPlatform, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpPlatform;
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform", err);
    }


    if (pID < platforms)
        currentplatform = tmpPlatform[pID];
    else
        currentplatform = tmpPlatform[0];

    delete[] tmpPlatform;

    //init device
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices);
    if (err != CL_SUCCESS || !devices)
    {
        throw ACException<ExceptionType::GPU, true>("Failed to find supported GPU", err);
    }

    cl_device_id* tmpDevice = new cl_device_id[devices];
    err = clGetDeviceIDs(currentplatform, CL_DEVICE_TYPE_GPU, devices, tmpDevice, nullptr);
    if (err != CL_SUCCESS)
    {
        delete[] tmpDevice;
        throw ACException<ExceptionType::GPU, true>("GPU initialization error", err);
    }

    if (dID < devices)
        device = tmpDevice[dID];
    else
        device = tmpDevice[0];

    delete[] tmpDevice;

    //init context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        releaseOpenCL();
        throw ACException<ExceptionType::GPU, true>("Failed to create context", err);
    }

    //init command queue
    commandQueueList.resize(commandQueueNum, nullptr);
#ifndef CL_VERSION_2_0 //for OpenCL SDK older than v2.0 to build
    for (int i = 0; i < commandQueueNum; i++)
    {
        commandQueueList[i] = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
    if (parallelIO)
    {
        commandQueueIO = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }

#else
    for (int i = 0; i < commandQueueNum; i++)
    {
#ifdef LEGACY_OPENCL_API
        commandQueueList[i] = clCreateCommandQueue(context, device, 0, &err);
#else
        commandQueueList[i] = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#endif
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
    if (parallelIO)
    {
#ifdef LEGACY_OPENCL_API
        commandQueueIO = clCreateCommandQueue(context, device, 0, &err);
#else
        commandQueueIO = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#endif
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create command queue", err);
        }
    }
#endif

#ifndef BUILT_IN_KERNEL
    //read kernel files
    std::string ACNetKernelSourceString[TotalTypeCount];
    std::string kernelFiles[TotalTypeCount] =
    { "ACNetKernel.cl", "ACNetHDNL1Kernel.cl" ,"ACNetHDNL2Kernel.cl" ,"ACNetHDNL3Kernel.cl" };
#endif // BUILT_IN_KERNEL
    const char* ACNetKernelSource[TotalTypeCount];

    cl_kernel tmpKernel = nullptr;
#ifdef ENABLE_FAST_MATH
    const char* buildFlags = "-cl-fast-relaxed-math";
#else
    const char* buildFlags = nullptr;
#endif // ENABLE_FAST_MATH
    switch (type)
    {
    case CNNType::ACNetHDNL0:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL0] = readKernel(kernelFiles[HDNL0]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL0] = ACNetKernelSourceString[HDNL0].c_str();

        //create program
        program[HDNL0] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL0], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL0], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL0], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL0], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL0], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::ACNetHDNL1:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL1] = readKernel(kernelFiles[HDNL1]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL1] = ACNetKernelSourceString[HDNL1].c_str();

        //create program
        program[HDNL1] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL1], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL1], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL1], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL1], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL1], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::ACNetHDNL2:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL2] = readKernel(kernelFiles[HDNL2]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL2] = ACNetKernelSourceString[HDNL2].c_str();

        //create program
        program[HDNL2] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL2], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL2], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL2], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL2], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL2], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::ACNetHDNL3:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        ACNetKernelSourceString[HDNL3] = readKernel(kernelFiles[HDNL3]);
#endif // BUILT_IN_KERNEL
        ACNetKernelSource[HDNL3] = ACNetKernelSourceString[HDNL3].c_str();

        //create program
        program[HDNL3] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[HDNL3], nullptr, &err);
        if (err != CL_SUCCESS)
        {
            releaseOpenCL();
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
        }

        //build program
        err = clBuildProgram(program[HDNL3], 1, &device, buildFlags, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t buildErrorSize = 0;
            clGetProgramBuildInfo(program[HDNL3], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
            char* buildError = new char[buildErrorSize];
            clGetProgramBuildInfo(program[HDNL3], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
            releaseOpenCL();
            ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
            delete[] buildError;
            throw exception;
        }

        tmpKernel = clCreateKernel(program[HDNL3], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    case CNNType::Default:
#ifndef BUILT_IN_KERNEL
        //read kernel files
        for (int i = HDNL0; i < TotalTypeCount; i++)
            ACNetKernelSourceString[i] = readKernel(kernelFiles[i]);
#endif // BUILT_IN_KERNEL
        for (int i = HDNL0; i < TotalTypeCount; i++)
        {
            ACNetKernelSource[i] = ACNetKernelSourceString[i].c_str();

            //create programACNet
            program[i] = clCreateProgramWithSource(context, 1, &ACNetKernelSource[i], nullptr, &err);
            if (err != CL_SUCCESS)
            {
                releaseOpenCL();
                throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL program", err);
            }

            //build programACNet
            err = clBuildProgram(program[i], 1, &device, buildFlags, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                size_t buildErrorSize = 0;
                clGetProgramBuildInfo(program[i], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildErrorSize);
                char* buildError = new char[buildErrorSize];
                clGetProgramBuildInfo(program[i], device, CL_PROGRAM_BUILD_LOG, buildErrorSize, buildError, nullptr);
                releaseOpenCL();
                ACException<ExceptionType::GPU, true> exception("Kernel build error", buildError, err);
                delete[] buildError;
                throw exception;
            }
        }

        tmpKernel = clCreateKernel(program[HDNL0], "conv8To8", &err);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to create OpenCL kernel for getting workGroupSizeLog", err);
        }
        err = clGetKernelWorkGroupInfo(tmpKernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), (void*)&workGroupSizeLog, nullptr);
        if (err != CL_SUCCESS)
        {
            clReleaseKernel(tmpKernel);
            throw ACException<ExceptionType::GPU, true>("Failed to get workGroupSize", err);
        }
        workGroupSizeLog = std::log2(workGroupSizeLog);
        break;
    }
    clReleaseKernel(tmpKernel);
}

void Anime4KCPP::OpenCL::ACNet::releaseOpenCL() noexcept
{
    for (auto& commandQueue : commandQueueList)
    {
        if (commandQueue != nullptr)
            clReleaseCommandQueue(commandQueue);
    }
    if (commandQueueIO != nullptr)
        clReleaseCommandQueue(commandQueueIO);
    for (int i = HDNL0; i < TotalTypeCount; i++)
    {
        if (program[i] != nullptr)
            clReleaseProgram(program[i]);
    }
    if (context != nullptr)
        clReleaseContext(context);
}

std::string Anime4KCPP::OpenCL::ACNet::readKernel(const std::string& fileName)
{
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open())
        throw ACException<ExceptionType::IO>("Failed to open kernel file.");

    std::ostringstream source;
    source << kernelFile.rdbuf();

    return source.str();
}

Anime4KCPP::Processor::Type Anime4KCPP::OpenCL::ACNet::getProcessorType() noexcept
{
    return Processor::Type::OpenCL_ACNet;
}

std::string Anime4KCPP::OpenCL::ACNet::getProcessorInfo()
{
    cl_int err = 0;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;

    size_t platformNameLength = 0;
    size_t deviceNameLength = 0;

    auto tmpPlatform = std::make_unique<cl_platform_id[]>(static_cast<size_t>(pID) + 1);
    err = clGetPlatformIDs(pID+1, tmpPlatform.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL platforms.", err);

    platform = tmpPlatform[pID];

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameLength);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform information.", err);

    auto platformName = std::make_unique<char[]>(platformNameLength);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameLength, platformName.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL platform information.", err);

    auto tmpDevice = std::make_unique<cl_device_id[]>(static_cast<size_t>(dID) + 1);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, dID + 1, tmpDevice.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to find OpenCL devices.", err);

    device = tmpDevice[dID];

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameLength);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL devices information.", err);

    auto deviceName = std::make_unique<char[]>(deviceNameLength);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameLength, deviceName.get(), nullptr);
    if (err != CL_SUCCESS)
        throw ACException<ExceptionType::GPU, true>("Failed to get OpenCL devices information.", err);

    std::ostringstream oss;
    oss << "Processor type: " << getProcessorType() << std::endl
        << "Current OpenCL devices:" << std::endl
        << " Platform " + std::to_string(pID) + ": " + platformName.get() << std::endl
        << "  Device " + std::to_string(dID) + ": " + deviceName.get();
    return oss.str();
}

//init OpenCL arguments
bool Anime4KCPP::OpenCL::ACNet::isInitialized = false;
cl_context Anime4KCPP::OpenCL::ACNet::context = nullptr;
int Anime4KCPP::OpenCL::ACNet::commandQueueNum = 4;
int Anime4KCPP::OpenCL::ACNet::commandQueueCount = 0;
std::vector<cl_command_queue> Anime4KCPP::OpenCL::ACNet::commandQueueList(commandQueueNum, nullptr);
bool Anime4KCPP::OpenCL::ACNet::parallelIO = false;
cl_command_queue Anime4KCPP::OpenCL::ACNet::commandQueueIO = nullptr;
cl_program Anime4KCPP::OpenCL::ACNet::program[TotalTypeCount] = { nullptr };
cl_device_id Anime4KCPP::OpenCL::ACNet::device = nullptr;
unsigned int Anime4KCPP::OpenCL::ACNet::pID = 0U;
unsigned int Anime4KCPP::OpenCL::ACNet::dID = 0U;
size_t Anime4KCPP::OpenCL::ACNet::workGroupSizeLog = 5;

#ifdef BUILT_IN_KERNEL
constexpr const char* kernelFunction =
R"(__kernel void conv1To8(
    __read_only image2d_t srcImg, 
    __write_only image2d_array_t tmpImgOut)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(srcImg) || y >= get_image_height(srcImg))
        return;
    
    const int2 imgIndex1 = (int2)(0, 0);
    const int2 imgIndex2 = (int2)(1, 0);
    int2 coord = (int2)(x, y);

    float4 tl = read_imagef(srcImg, samplerN, (int2)(x-1, y-1));
    float4 tc = read_imagef(srcImg, samplerN, (int2)(x, y-1));
    float4 tr = read_imagef(srcImg, samplerN, (int2)(x+1, y-1));
    float4 ml = read_imagef(srcImg, samplerN, (int2)(x-1, y));
    float4 mc = read_imagef(srcImg, samplerN, coord);
    float4 mr = read_imagef(srcImg, samplerN, (int2)(x+1, y));
    float4 bl = read_imagef(srcImg, samplerN, (int2)(x-1, y+1));
    float4 bc = read_imagef(srcImg, samplerN, (int2)(x, y+1));
    float4 br = read_imagef(srcImg, samplerN, (int2)(x+1, y+1));

    float4 c1234 = RELU((float4)(
        tl.x * kernelsL1[0*9+0] + tc.x * kernelsL1[0*9+1] + tr.x * kernelsL1[0*9+2] +
        ml.x * kernelsL1[0*9+3] + mc.x * kernelsL1[0*9+4] + mr.x * kernelsL1[0*9+5] +
        bl.x * kernelsL1[0*9+6] + bc.x * kernelsL1[0*9+7] + br.x * kernelsL1[0*9+8] + biasL1[0],

        tl.x * kernelsL1[1*9+0] + tc.x * kernelsL1[1*9+1] + tr.x * kernelsL1[1*9+2] +
        ml.x * kernelsL1[1*9+3] + mc.x * kernelsL1[1*9+4] + mr.x * kernelsL1[1*9+5] +
        bl.x * kernelsL1[1*9+6] + bc.x * kernelsL1[1*9+7] + br.x * kernelsL1[1*9+8] + biasL1[1],

        tl.x * kernelsL1[2*9+0] + tc.x * kernelsL1[2*9+1] + tr.x * kernelsL1[2*9+2] +
        ml.x * kernelsL1[2*9+3] + mc.x * kernelsL1[2*9+4] + mr.x * kernelsL1[2*9+5] +
        bl.x * kernelsL1[2*9+6] + bc.x * kernelsL1[2*9+7] + br.x * kernelsL1[2*9+8] + biasL1[2],

        tl.x * kernelsL1[3*9+0] + tc.x * kernelsL1[3*9+1] + tr.x * kernelsL1[3*9+2] +
        ml.x * kernelsL1[3*9+3] + mc.x * kernelsL1[3*9+4] + mr.x * kernelsL1[3*9+5] +
        bl.x * kernelsL1[3*9+6] + bc.x * kernelsL1[3*9+7] + br.x * kernelsL1[3*9+8] + biasL1[3]
    ));
    float4 c5678 = RELU((float4)(
        tl.x * kernelsL1[4*9+0] + tc.x * kernelsL1[4*9+1] + tr.x * kernelsL1[4*9+2] +
        ml.x * kernelsL1[4*9+3] + mc.x * kernelsL1[4*9+4] + mr.x * kernelsL1[4*9+5] +
        bl.x * kernelsL1[4*9+6] + bc.x * kernelsL1[4*9+7] + br.x * kernelsL1[4*9+8] + biasL1[4],

        tl.x * kernelsL1[5*9+0] + tc.x * kernelsL1[5*9+1] + tr.x * kernelsL1[5*9+2] +
        ml.x * kernelsL1[5*9+3] + mc.x * kernelsL1[5*9+4] + mr.x * kernelsL1[5*9+5] +
        bl.x * kernelsL1[5*9+6] + bc.x * kernelsL1[5*9+7] + br.x * kernelsL1[5*9+8] + biasL1[5],

        tl.x * kernelsL1[6*9+0] + tc.x * kernelsL1[6*9+1] + tr.x * kernelsL1[6*9+2] +
        ml.x * kernelsL1[6*9+3] + mc.x * kernelsL1[6*9+4] + mr.x * kernelsL1[6*9+5] +
        bl.x * kernelsL1[6*9+6] + bc.x * kernelsL1[6*9+7] + br.x * kernelsL1[6*9+8] + biasL1[6],

        tl.x * kernelsL1[7*9+0] + tc.x * kernelsL1[7*9+1] + tr.x * kernelsL1[7*9+2] +
        ml.x * kernelsL1[7*9+3] + mc.x * kernelsL1[7*9+4] + mr.x * kernelsL1[7*9+5] +
        bl.x * kernelsL1[7*9+6] + bc.x * kernelsL1[7*9+7] + br.x * kernelsL1[7*9+8] + biasL1[7]
    ));

    write_imagef(tmpImgOut, (int4)(coord, imgIndex1), c1234);
    write_imagef(tmpImgOut, (int4)(coord, imgIndex2), c5678);
}

__kernel void conv8To8(
    __read_only image2d_array_t tmpImgIn,
    __write_only image2d_array_t tmpImgOut,
    int l)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(tmpImgIn) || y >= get_image_height(tmpImgIn))
        return;

    const int2 imgIndex1 = (int2)(0, 0);
    const int2 imgIndex2 = (int2)(1, 0);
    int4 coord1 = (int4)(x, y, imgIndex1);
    int4 coord2 = (int4)(x, y, imgIndex2);

    float4 tl1 = read_imagef(tmpImgIn, samplerN, (int4)(x-1, y-1, imgIndex1));
    float4 tc1 = read_imagef(tmpImgIn, samplerN, (int4)(x, y-1, imgIndex1));
    float4 tr1 = read_imagef(tmpImgIn, samplerN, (int4)(x+1, y-1, imgIndex1));
    float4 ml1 = read_imagef(tmpImgIn, samplerN, (int4)(x-1, y, imgIndex1));
    float4 mc1 = read_imagef(tmpImgIn, samplerN, coord1);
    float4 mr1 = read_imagef(tmpImgIn, samplerN, (int4)(x+1, y, imgIndex1));
    float4 bl1 = read_imagef(tmpImgIn, samplerN, (int4)(x-1, y+1, imgIndex1));
    float4 bc1 = read_imagef(tmpImgIn, samplerN, (int4)(x, y+1, imgIndex1));
    float4 br1 = read_imagef(tmpImgIn, samplerN, (int4)(x+1, y+1, imgIndex1));

    float4 tl2 = read_imagef(tmpImgIn, samplerN, (int4)(x-1, y-1, imgIndex2));
    float4 tc2 = read_imagef(tmpImgIn, samplerN, (int4)(x, y-1, imgIndex2));
    float4 tr2 = read_imagef(tmpImgIn, samplerN, (int4)(x+1, y-1, imgIndex2));
    float4 ml2 = read_imagef(tmpImgIn, samplerN, (int4)(x-1, y, imgIndex2));
    float4 mc2 = read_imagef(tmpImgIn, samplerN, coord2);
    float4 mr2 = read_imagef(tmpImgIn, samplerN, (int4)(x+1, y, imgIndex2));
    float4 bl2 = read_imagef(tmpImgIn, samplerN, (int4)(x-1, y+1, imgIndex2));
    float4 bc2 = read_imagef(tmpImgIn, samplerN, (int4)(x, y+1, imgIndex2));
    float4 br2 = read_imagef(tmpImgIn, samplerN, (int4)(x+1, y+1, imgIndex2));
)"
R"(    
    float4 c1234 = RELU((float4)(
        tl1.x * kernelsL[l][0*72+0*9+0] + tc1.x * kernelsL[l][0*72+0*9+1] + tr1.x * kernelsL[l][0*72+0*9+2] +
        ml1.x * kernelsL[l][0*72+0*9+3] + mc1.x * kernelsL[l][0*72+0*9+4] + mr1.x * kernelsL[l][0*72+0*9+5] +
        bl1.x * kernelsL[l][0*72+0*9+6] + bc1.x * kernelsL[l][0*72+0*9+7] + br1.x * kernelsL[l][0*72+0*9+8] + 

        tl1.y * kernelsL[l][0*72+1*9+0] + tc1.y * kernelsL[l][0*72+1*9+1] + tr1.y * kernelsL[l][0*72+1*9+2] +
        ml1.y * kernelsL[l][0*72+1*9+3] + mc1.y * kernelsL[l][0*72+1*9+4] + mr1.y * kernelsL[l][0*72+1*9+5] +
        bl1.y * kernelsL[l][0*72+1*9+6] + bc1.y * kernelsL[l][0*72+1*9+7] + br1.y * kernelsL[l][0*72+1*9+8] + 

        tl1.z * kernelsL[l][0*72+2*9+0] + tc1.z * kernelsL[l][0*72+2*9+1] + tr1.z * kernelsL[l][0*72+2*9+2] +
        ml1.z * kernelsL[l][0*72+2*9+3] + mc1.z * kernelsL[l][0*72+2*9+4] + mr1.z * kernelsL[l][0*72+2*9+5] +
        bl1.z * kernelsL[l][0*72+2*9+6] + bc1.z * kernelsL[l][0*72+2*9+7] + br1.z * kernelsL[l][0*72+2*9+8] + 

        tl1.w * kernelsL[l][0*72+3*9+0] + tc1.w * kernelsL[l][0*72+3*9+1] + tr1.w * kernelsL[l][0*72+3*9+2] +
        ml1.w * kernelsL[l][0*72+3*9+3] + mc1.w * kernelsL[l][0*72+3*9+4] + mr1.w * kernelsL[l][0*72+3*9+5] +
        bl1.w * kernelsL[l][0*72+3*9+6] + bc1.w * kernelsL[l][0*72+3*9+7] + br1.w * kernelsL[l][0*72+3*9+8] +

        tl2.x * kernelsL[l][0*72+4*9+0] + tc2.x * kernelsL[l][0*72+4*9+1] + tr2.x * kernelsL[l][0*72+4*9+2] +
        ml2.x * kernelsL[l][0*72+4*9+3] + mc2.x * kernelsL[l][0*72+4*9+4] + mr2.x * kernelsL[l][0*72+4*9+5] +
        bl2.x * kernelsL[l][0*72+4*9+6] + bc2.x * kernelsL[l][0*72+4*9+7] + br2.x * kernelsL[l][0*72+4*9+8] + 

        tl2.y * kernelsL[l][0*72+5*9+0] + tc2.y * kernelsL[l][0*72+5*9+1] + tr2.y * kernelsL[l][0*72+5*9+2] +
        ml2.y * kernelsL[l][0*72+5*9+3] + mc2.y * kernelsL[l][0*72+5*9+4] + mr2.y * kernelsL[l][0*72+5*9+5] +
        bl2.y * kernelsL[l][0*72+5*9+6] + bc2.y * kernelsL[l][0*72+5*9+7] + br2.y * kernelsL[l][0*72+5*9+8] + 

        tl2.z * kernelsL[l][0*72+6*9+0] + tc2.z * kernelsL[l][0*72+6*9+1] + tr2.z * kernelsL[l][0*72+6*9+2] +
        ml2.z * kernelsL[l][0*72+6*9+3] + mc2.z * kernelsL[l][0*72+6*9+4] + mr2.z * kernelsL[l][0*72+6*9+5] +
        bl2.z * kernelsL[l][0*72+6*9+6] + bc2.z * kernelsL[l][0*72+6*9+7] + br2.z * kernelsL[l][0*72+6*9+8] + 

        tl2.w * kernelsL[l][0*72+7*9+0] + tc2.w * kernelsL[l][0*72+7*9+1] + tr2.w * kernelsL[l][0*72+7*9+2] +
        ml2.w * kernelsL[l][0*72+7*9+3] + mc2.w * kernelsL[l][0*72+7*9+4] + mr2.w * kernelsL[l][0*72+7*9+5] +
        bl2.w * kernelsL[l][0*72+7*9+6] + bc2.w * kernelsL[l][0*72+7*9+7] + br2.w * kernelsL[l][0*72+7*9+8] + biasL[l][0]
        ,
        tl1.x * kernelsL[l][1*72+0*9+0] + tc1.x * kernelsL[l][1*72+0*9+1] + tr1.x * kernelsL[l][1*72+0*9+2] +
        ml1.x * kernelsL[l][1*72+0*9+3] + mc1.x * kernelsL[l][1*72+0*9+4] + mr1.x * kernelsL[l][1*72+0*9+5] +
        bl1.x * kernelsL[l][1*72+0*9+6] + bc1.x * kernelsL[l][1*72+0*9+7] + br1.x * kernelsL[l][1*72+0*9+8] + 

        tl1.y * kernelsL[l][1*72+1*9+0] + tc1.y * kernelsL[l][1*72+1*9+1] + tr1.y * kernelsL[l][1*72+1*9+2] +
        ml1.y * kernelsL[l][1*72+1*9+3] + mc1.y * kernelsL[l][1*72+1*9+4] + mr1.y * kernelsL[l][1*72+1*9+5] +
        bl1.y * kernelsL[l][1*72+1*9+6] + bc1.y * kernelsL[l][1*72+1*9+7] + br1.y * kernelsL[l][1*72+1*9+8] + 

        tl1.z * kernelsL[l][1*72+2*9+0] + tc1.z * kernelsL[l][1*72+2*9+1] + tr1.z * kernelsL[l][1*72+2*9+2] +
        ml1.z * kernelsL[l][1*72+2*9+3] + mc1.z * kernelsL[l][1*72+2*9+4] + mr1.z * kernelsL[l][1*72+2*9+5] +
        bl1.z * kernelsL[l][1*72+2*9+6] + bc1.z * kernelsL[l][1*72+2*9+7] + br1.z * kernelsL[l][1*72+2*9+8] + 

        tl1.w * kernelsL[l][1*72+3*9+0] + tc1.w * kernelsL[l][1*72+3*9+1] + tr1.w * kernelsL[l][1*72+3*9+2] +
        ml1.w * kernelsL[l][1*72+3*9+3] + mc1.w * kernelsL[l][1*72+3*9+4] + mr1.w * kernelsL[l][1*72+3*9+5] +
        bl1.w * kernelsL[l][1*72+3*9+6] + bc1.w * kernelsL[l][1*72+3*9+7] + br1.w * kernelsL[l][1*72+3*9+8] +

        tl2.x * kernelsL[l][1*72+4*9+0] + tc2.x * kernelsL[l][1*72+4*9+1] + tr2.x * kernelsL[l][1*72+4*9+2] +
        ml2.x * kernelsL[l][1*72+4*9+3] + mc2.x * kernelsL[l][1*72+4*9+4] + mr2.x * kernelsL[l][1*72+4*9+5] +
        bl2.x * kernelsL[l][1*72+4*9+6] + bc2.x * kernelsL[l][1*72+4*9+7] + br2.x * kernelsL[l][1*72+4*9+8] + 

        tl2.y * kernelsL[l][1*72+5*9+0] + tc2.y * kernelsL[l][1*72+5*9+1] + tr2.y * kernelsL[l][1*72+5*9+2] +
        ml2.y * kernelsL[l][1*72+5*9+3] + mc2.y * kernelsL[l][1*72+5*9+4] + mr2.y * kernelsL[l][1*72+5*9+5] +
        bl2.y * kernelsL[l][1*72+5*9+6] + bc2.y * kernelsL[l][1*72+5*9+7] + br2.y * kernelsL[l][1*72+5*9+8] + 

        tl2.z * kernelsL[l][1*72+6*9+0] + tc2.z * kernelsL[l][1*72+6*9+1] + tr2.z * kernelsL[l][1*72+6*9+2] +
        ml2.z * kernelsL[l][1*72+6*9+3] + mc2.z * kernelsL[l][1*72+6*9+4] + mr2.z * kernelsL[l][1*72+6*9+5] +
        bl2.z * kernelsL[l][1*72+6*9+6] + bc2.z * kernelsL[l][1*72+6*9+7] + br2.z * kernelsL[l][1*72+6*9+8] + 

        tl2.w * kernelsL[l][1*72+7*9+0] + tc2.w * kernelsL[l][1*72+7*9+1] + tr2.w * kernelsL[l][1*72+7*9+2] +
        ml2.w * kernelsL[l][1*72+7*9+3] + mc2.w * kernelsL[l][1*72+7*9+4] + mr2.w * kernelsL[l][1*72+7*9+5] +
        bl2.w * kernelsL[l][1*72+7*9+6] + bc2.w * kernelsL[l][1*72+7*9+7] + br2.w * kernelsL[l][1*72+7*9+8] + biasL[l][1]
        ,
        tl1.x * kernelsL[l][2*72+0*9+0] + tc1.x * kernelsL[l][2*72+0*9+1] + tr1.x * kernelsL[l][2*72+0*9+2] +
        ml1.x * kernelsL[l][2*72+0*9+3] + mc1.x * kernelsL[l][2*72+0*9+4] + mr1.x * kernelsL[l][2*72+0*9+5] +
        bl1.x * kernelsL[l][2*72+0*9+6] + bc1.x * kernelsL[l][2*72+0*9+7] + br1.x * kernelsL[l][2*72+0*9+8] + 

        tl1.y * kernelsL[l][2*72+1*9+0] + tc1.y * kernelsL[l][2*72+1*9+1] + tr1.y * kernelsL[l][2*72+1*9+2] +
        ml1.y * kernelsL[l][2*72+1*9+3] + mc1.y * kernelsL[l][2*72+1*9+4] + mr1.y * kernelsL[l][2*72+1*9+5] +
        bl1.y * kernelsL[l][2*72+1*9+6] + bc1.y * kernelsL[l][2*72+1*9+7] + br1.y * kernelsL[l][2*72+1*9+8] + 

        tl1.z * kernelsL[l][2*72+2*9+0] + tc1.z * kernelsL[l][2*72+2*9+1] + tr1.z * kernelsL[l][2*72+2*9+2] +
        ml1.z * kernelsL[l][2*72+2*9+3] + mc1.z * kernelsL[l][2*72+2*9+4] + mr1.z * kernelsL[l][2*72+2*9+5] +
        bl1.z * kernelsL[l][2*72+2*9+6] + bc1.z * kernelsL[l][2*72+2*9+7] + br1.z * kernelsL[l][2*72+2*9+8] + 

        tl1.w * kernelsL[l][2*72+3*9+0] + tc1.w * kernelsL[l][2*72+3*9+1] + tr1.w * kernelsL[l][2*72+3*9+2] +
        ml1.w * kernelsL[l][2*72+3*9+3] + mc1.w * kernelsL[l][2*72+3*9+4] + mr1.w * kernelsL[l][2*72+3*9+5] +
        bl1.w * kernelsL[l][2*72+3*9+6] + bc1.w * kernelsL[l][2*72+3*9+7] + br1.w * kernelsL[l][2*72+3*9+8] +

        tl2.x * kernelsL[l][2*72+4*9+0] + tc2.x * kernelsL[l][2*72+4*9+1] + tr2.x * kernelsL[l][2*72+4*9+2] +
        ml2.x * kernelsL[l][2*72+4*9+3] + mc2.x * kernelsL[l][2*72+4*9+4] + mr2.x * kernelsL[l][2*72+4*9+5] +
        bl2.x * kernelsL[l][2*72+4*9+6] + bc2.x * kernelsL[l][2*72+4*9+7] + br2.x * kernelsL[l][2*72+4*9+8] + 

        tl2.y * kernelsL[l][2*72+5*9+0] + tc2.y * kernelsL[l][2*72+5*9+1] + tr2.y * kernelsL[l][2*72+5*9+2] +
        ml2.y * kernelsL[l][2*72+5*9+3] + mc2.y * kernelsL[l][2*72+5*9+4] + mr2.y * kernelsL[l][2*72+5*9+5] +
        bl2.y * kernelsL[l][2*72+5*9+6] + bc2.y * kernelsL[l][2*72+5*9+7] + br2.y * kernelsL[l][2*72+5*9+8] + 

        tl2.z * kernelsL[l][2*72+6*9+0] + tc2.z * kernelsL[l][2*72+6*9+1] + tr2.z * kernelsL[l][2*72+6*9+2] +
        ml2.z * kernelsL[l][2*72+6*9+3] + mc2.z * kernelsL[l][2*72+6*9+4] + mr2.z * kernelsL[l][2*72+6*9+5] +
        bl2.z * kernelsL[l][2*72+6*9+6] + bc2.z * kernelsL[l][2*72+6*9+7] + br2.z * kernelsL[l][2*72+6*9+8] + 

        tl2.w * kernelsL[l][2*72+7*9+0] + tc2.w * kernelsL[l][2*72+7*9+1] + tr2.w * kernelsL[l][2*72+7*9+2] +
        ml2.w * kernelsL[l][2*72+7*9+3] + mc2.w * kernelsL[l][2*72+7*9+4] + mr2.w * kernelsL[l][2*72+7*9+5] +
        bl2.w * kernelsL[l][2*72+7*9+6] + bc2.w * kernelsL[l][2*72+7*9+7] + br2.w * kernelsL[l][2*72+7*9+8] + biasL[l][2]
        ,
        tl1.x * kernelsL[l][3*72+0*9+0] + tc1.x * kernelsL[l][3*72+0*9+1] + tr1.x * kernelsL[l][3*72+0*9+2] +
        ml1.x * kernelsL[l][3*72+0*9+3] + mc1.x * kernelsL[l][3*72+0*9+4] + mr1.x * kernelsL[l][3*72+0*9+5] +
        bl1.x * kernelsL[l][3*72+0*9+6] + bc1.x * kernelsL[l][3*72+0*9+7] + br1.x * kernelsL[l][3*72+0*9+8] + 
)"
R"(
        tl1.y * kernelsL[l][3*72+1*9+0] + tc1.y * kernelsL[l][3*72+1*9+1] + tr1.y * kernelsL[l][3*72+1*9+2] +
        ml1.y * kernelsL[l][3*72+1*9+3] + mc1.y * kernelsL[l][3*72+1*9+4] + mr1.y * kernelsL[l][3*72+1*9+5] +
        bl1.y * kernelsL[l][3*72+1*9+6] + bc1.y * kernelsL[l][3*72+1*9+7] + br1.y * kernelsL[l][3*72+1*9+8] + 

        tl1.z * kernelsL[l][3*72+2*9+0] + tc1.z * kernelsL[l][3*72+2*9+1] + tr1.z * kernelsL[l][3*72+2*9+2] +
        ml1.z * kernelsL[l][3*72+2*9+3] + mc1.z * kernelsL[l][3*72+2*9+4] + mr1.z * kernelsL[l][3*72+2*9+5] +
        bl1.z * kernelsL[l][3*72+2*9+6] + bc1.z * kernelsL[l][3*72+2*9+7] + br1.z * kernelsL[l][3*72+2*9+8] + 

        tl1.w * kernelsL[l][3*72+3*9+0] + tc1.w * kernelsL[l][3*72+3*9+1] + tr1.w * kernelsL[l][3*72+3*9+2] +
        ml1.w * kernelsL[l][3*72+3*9+3] + mc1.w * kernelsL[l][3*72+3*9+4] + mr1.w * kernelsL[l][3*72+3*9+5] +
        bl1.w * kernelsL[l][3*72+3*9+6] + bc1.w * kernelsL[l][3*72+3*9+7] + br1.w * kernelsL[l][3*72+3*9+8] +

        tl2.x * kernelsL[l][3*72+4*9+0] + tc2.x * kernelsL[l][3*72+4*9+1] + tr2.x * kernelsL[l][3*72+4*9+2] +
        ml2.x * kernelsL[l][3*72+4*9+3] + mc2.x * kernelsL[l][3*72+4*9+4] + mr2.x * kernelsL[l][3*72+4*9+5] +
        bl2.x * kernelsL[l][3*72+4*9+6] + bc2.x * kernelsL[l][3*72+4*9+7] + br2.x * kernelsL[l][3*72+4*9+8] + 

        tl2.y * kernelsL[l][3*72+5*9+0] + tc2.y * kernelsL[l][3*72+5*9+1] + tr2.y * kernelsL[l][3*72+5*9+2] +
        ml2.y * kernelsL[l][3*72+5*9+3] + mc2.y * kernelsL[l][3*72+5*9+4] + mr2.y * kernelsL[l][3*72+5*9+5] +
        bl2.y * kernelsL[l][3*72+5*9+6] + bc2.y * kernelsL[l][3*72+5*9+7] + br2.y * kernelsL[l][3*72+5*9+8] + 

        tl2.z * kernelsL[l][3*72+6*9+0] + tc2.z * kernelsL[l][3*72+6*9+1] + tr2.z * kernelsL[l][3*72+6*9+2] +
        ml2.z * kernelsL[l][3*72+6*9+3] + mc2.z * kernelsL[l][3*72+6*9+4] + mr2.z * kernelsL[l][3*72+6*9+5] +
        bl2.z * kernelsL[l][3*72+6*9+6] + bc2.z * kernelsL[l][3*72+6*9+7] + br2.z * kernelsL[l][3*72+6*9+8] + 

        tl2.w * kernelsL[l][3*72+7*9+0] + tc2.w * kernelsL[l][3*72+7*9+1] + tr2.w * kernelsL[l][3*72+7*9+2] +
        ml2.w * kernelsL[l][3*72+7*9+3] + mc2.w * kernelsL[l][3*72+7*9+4] + mr2.w * kernelsL[l][3*72+7*9+5] +
        bl2.w * kernelsL[l][3*72+7*9+6] + bc2.w * kernelsL[l][3*72+7*9+7] + br2.w * kernelsL[l][3*72+7*9+8] + biasL[l][3]
    ));
    float4 c5678 = RELU((float4)(
        tl1.x * kernelsL[l][4*72+0*9+0] + tc1.x * kernelsL[l][4*72+0*9+1] + tr1.x * kernelsL[l][4*72+0*9+2] +
        ml1.x * kernelsL[l][4*72+0*9+3] + mc1.x * kernelsL[l][4*72+0*9+4] + mr1.x * kernelsL[l][4*72+0*9+5] +
        bl1.x * kernelsL[l][4*72+0*9+6] + bc1.x * kernelsL[l][4*72+0*9+7] + br1.x * kernelsL[l][4*72+0*9+8] + 

        tl1.y * kernelsL[l][4*72+1*9+0] + tc1.y * kernelsL[l][4*72+1*9+1] + tr1.y * kernelsL[l][4*72+1*9+2] +
        ml1.y * kernelsL[l][4*72+1*9+3] + mc1.y * kernelsL[l][4*72+1*9+4] + mr1.y * kernelsL[l][4*72+1*9+5] +
        bl1.y * kernelsL[l][4*72+1*9+6] + bc1.y * kernelsL[l][4*72+1*9+7] + br1.y * kernelsL[l][4*72+1*9+8] + 

        tl1.z * kernelsL[l][4*72+2*9+0] + tc1.z * kernelsL[l][4*72+2*9+1] + tr1.z * kernelsL[l][4*72+2*9+2] +
        ml1.z * kernelsL[l][4*72+2*9+3] + mc1.z * kernelsL[l][4*72+2*9+4] + mr1.z * kernelsL[l][4*72+2*9+5] +
        bl1.z * kernelsL[l][4*72+2*9+6] + bc1.z * kernelsL[l][4*72+2*9+7] + br1.z * kernelsL[l][4*72+2*9+8] + 

        tl1.w * kernelsL[l][4*72+3*9+0] + tc1.w * kernelsL[l][4*72+3*9+1] + tr1.w * kernelsL[l][4*72+3*9+2] +
        ml1.w * kernelsL[l][4*72+3*9+3] + mc1.w * kernelsL[l][4*72+3*9+4] + mr1.w * kernelsL[l][4*72+3*9+5] +
        bl1.w * kernelsL[l][4*72+3*9+6] + bc1.w * kernelsL[l][4*72+3*9+7] + br1.w * kernelsL[l][4*72+3*9+8] +

        tl2.x * kernelsL[l][4*72+4*9+0] + tc2.x * kernelsL[l][4*72+4*9+1] + tr2.x * kernelsL[l][4*72+4*9+2] +
        ml2.x * kernelsL[l][4*72+4*9+3] + mc2.x * kernelsL[l][4*72+4*9+4] + mr2.x * kernelsL[l][4*72+4*9+5] +
        bl2.x * kernelsL[l][4*72+4*9+6] + bc2.x * kernelsL[l][4*72+4*9+7] + br2.x * kernelsL[l][4*72+4*9+8] + 

        tl2.y * kernelsL[l][4*72+5*9+0] + tc2.y * kernelsL[l][4*72+5*9+1] + tr2.y * kernelsL[l][4*72+5*9+2] +
        ml2.y * kernelsL[l][4*72+5*9+3] + mc2.y * kernelsL[l][4*72+5*9+4] + mr2.y * kernelsL[l][4*72+5*9+5] +
        bl2.y * kernelsL[l][4*72+5*9+6] + bc2.y * kernelsL[l][4*72+5*9+7] + br2.y * kernelsL[l][4*72+5*9+8] + 

        tl2.z * kernelsL[l][4*72+6*9+0] + tc2.z * kernelsL[l][4*72+6*9+1] + tr2.z * kernelsL[l][4*72+6*9+2] +
        ml2.z * kernelsL[l][4*72+6*9+3] + mc2.z * kernelsL[l][4*72+6*9+4] + mr2.z * kernelsL[l][4*72+6*9+5] +
        bl2.z * kernelsL[l][4*72+6*9+6] + bc2.z * kernelsL[l][4*72+6*9+7] + br2.z * kernelsL[l][4*72+6*9+8] + 

        tl2.w * kernelsL[l][4*72+7*9+0] + tc2.w * kernelsL[l][4*72+7*9+1] + tr2.w * kernelsL[l][4*72+7*9+2] +
        ml2.w * kernelsL[l][4*72+7*9+3] + mc2.w * kernelsL[l][4*72+7*9+4] + mr2.w * kernelsL[l][4*72+7*9+5] +
        bl2.w * kernelsL[l][4*72+7*9+6] + bc2.w * kernelsL[l][4*72+7*9+7] + br2.w * kernelsL[l][4*72+7*9+8] + biasL[l][4]
        ,
        tl1.x * kernelsL[l][5*72+0*9+0] + tc1.x * kernelsL[l][5*72+0*9+1] + tr1.x * kernelsL[l][5*72+0*9+2] +
        ml1.x * kernelsL[l][5*72+0*9+3] + mc1.x * kernelsL[l][5*72+0*9+4] + mr1.x * kernelsL[l][5*72+0*9+5] +
        bl1.x * kernelsL[l][5*72+0*9+6] + bc1.x * kernelsL[l][5*72+0*9+7] + br1.x * kernelsL[l][5*72+0*9+8] + 

        tl1.y * kernelsL[l][5*72+1*9+0] + tc1.y * kernelsL[l][5*72+1*9+1] + tr1.y * kernelsL[l][5*72+1*9+2] +
        ml1.y * kernelsL[l][5*72+1*9+3] + mc1.y * kernelsL[l][5*72+1*9+4] + mr1.y * kernelsL[l][5*72+1*9+5] +
        bl1.y * kernelsL[l][5*72+1*9+6] + bc1.y * kernelsL[l][5*72+1*9+7] + br1.y * kernelsL[l][5*72+1*9+8] + 

        tl1.z * kernelsL[l][5*72+2*9+0] + tc1.z * kernelsL[l][5*72+2*9+1] + tr1.z * kernelsL[l][5*72+2*9+2] +
        ml1.z * kernelsL[l][5*72+2*9+3] + mc1.z * kernelsL[l][5*72+2*9+4] + mr1.z * kernelsL[l][5*72+2*9+5] +
        bl1.z * kernelsL[l][5*72+2*9+6] + bc1.z * kernelsL[l][5*72+2*9+7] + br1.z * kernelsL[l][5*72+2*9+8] + 

        tl1.w * kernelsL[l][5*72+3*9+0] + tc1.w * kernelsL[l][5*72+3*9+1] + tr1.w * kernelsL[l][5*72+3*9+2] +
        ml1.w * kernelsL[l][5*72+3*9+3] + mc1.w * kernelsL[l][5*72+3*9+4] + mr1.w * kernelsL[l][5*72+3*9+5] +
        bl1.w * kernelsL[l][5*72+3*9+6] + bc1.w * kernelsL[l][5*72+3*9+7] + br1.w * kernelsL[l][5*72+3*9+8] +

        tl2.x * kernelsL[l][5*72+4*9+0] + tc2.x * kernelsL[l][5*72+4*9+1] + tr2.x * kernelsL[l][5*72+4*9+2] +
        ml2.x * kernelsL[l][5*72+4*9+3] + mc2.x * kernelsL[l][5*72+4*9+4] + mr2.x * kernelsL[l][5*72+4*9+5] +
        bl2.x * kernelsL[l][5*72+4*9+6] + bc2.x * kernelsL[l][5*72+4*9+7] + br2.x * kernelsL[l][5*72+4*9+8] + 

        tl2.y * kernelsL[l][5*72+5*9+0] + tc2.y * kernelsL[l][5*72+5*9+1] + tr2.y * kernelsL[l][5*72+5*9+2] +
        ml2.y * kernelsL[l][5*72+5*9+3] + mc2.y * kernelsL[l][5*72+5*9+4] + mr2.y * kernelsL[l][5*72+5*9+5] +
        bl2.y * kernelsL[l][5*72+5*9+6] + bc2.y * kernelsL[l][5*72+5*9+7] + br2.y * kernelsL[l][5*72+5*9+8] + 

        tl2.z * kernelsL[l][5*72+6*9+0] + tc2.z * kernelsL[l][5*72+6*9+1] + tr2.z * kernelsL[l][5*72+6*9+2] +
        ml2.z * kernelsL[l][5*72+6*9+3] + mc2.z * kernelsL[l][5*72+6*9+4] + mr2.z * kernelsL[l][5*72+6*9+5] +
        bl2.z * kernelsL[l][5*72+6*9+6] + bc2.z * kernelsL[l][5*72+6*9+7] + br2.z * kernelsL[l][5*72+6*9+8] + 

        tl2.w * kernelsL[l][5*72+7*9+0] + tc2.w * kernelsL[l][5*72+7*9+1] + tr2.w * kernelsL[l][5*72+7*9+2] +
        ml2.w * kernelsL[l][5*72+7*9+3] + mc2.w * kernelsL[l][5*72+7*9+4] + mr2.w * kernelsL[l][5*72+7*9+5] +
        bl2.w * kernelsL[l][5*72+7*9+6] + bc2.w * kernelsL[l][5*72+7*9+7] + br2.w * kernelsL[l][5*72+7*9+8] + biasL[l][5]
        ,
        tl1.x * kernelsL[l][6*72+0*9+0] + tc1.x * kernelsL[l][6*72+0*9+1] + tr1.x * kernelsL[l][6*72+0*9+2] +
        ml1.x * kernelsL[l][6*72+0*9+3] + mc1.x * kernelsL[l][6*72+0*9+4] + mr1.x * kernelsL[l][6*72+0*9+5] +
        bl1.x * kernelsL[l][6*72+0*9+6] + bc1.x * kernelsL[l][6*72+0*9+7] + br1.x * kernelsL[l][6*72+0*9+8] + 

        tl1.y * kernelsL[l][6*72+1*9+0] + tc1.y * kernelsL[l][6*72+1*9+1] + tr1.y * kernelsL[l][6*72+1*9+2] +
        ml1.y * kernelsL[l][6*72+1*9+3] + mc1.y * kernelsL[l][6*72+1*9+4] + mr1.y * kernelsL[l][6*72+1*9+5] +
        bl1.y * kernelsL[l][6*72+1*9+6] + bc1.y * kernelsL[l][6*72+1*9+7] + br1.y * kernelsL[l][6*72+1*9+8] + 

        tl1.z * kernelsL[l][6*72+2*9+0] + tc1.z * kernelsL[l][6*72+2*9+1] + tr1.z * kernelsL[l][6*72+2*9+2] +
        ml1.z * kernelsL[l][6*72+2*9+3] + mc1.z * kernelsL[l][6*72+2*9+4] + mr1.z * kernelsL[l][6*72+2*9+5] +
        bl1.z * kernelsL[l][6*72+2*9+6] + bc1.z * kernelsL[l][6*72+2*9+7] + br1.z * kernelsL[l][6*72+2*9+8] + 

        tl1.w * kernelsL[l][6*72+3*9+0] + tc1.w * kernelsL[l][6*72+3*9+1] + tr1.w * kernelsL[l][6*72+3*9+2] +
        ml1.w * kernelsL[l][6*72+3*9+3] + mc1.w * kernelsL[l][6*72+3*9+4] + mr1.w * kernelsL[l][6*72+3*9+5] +
        bl1.w * kernelsL[l][6*72+3*9+6] + bc1.w * kernelsL[l][6*72+3*9+7] + br1.w * kernelsL[l][6*72+3*9+8] +

        tl2.x * kernelsL[l][6*72+4*9+0] + tc2.x * kernelsL[l][6*72+4*9+1] + tr2.x * kernelsL[l][6*72+4*9+2] +
        ml2.x * kernelsL[l][6*72+4*9+3] + mc2.x * kernelsL[l][6*72+4*9+4] + mr2.x * kernelsL[l][6*72+4*9+5] +
        bl2.x * kernelsL[l][6*72+4*9+6] + bc2.x * kernelsL[l][6*72+4*9+7] + br2.x * kernelsL[l][6*72+4*9+8] + 

        tl2.y * kernelsL[l][6*72+5*9+0] + tc2.y * kernelsL[l][6*72+5*9+1] + tr2.y * kernelsL[l][6*72+5*9+2] +
        ml2.y * kernelsL[l][6*72+5*9+3] + mc2.y * kernelsL[l][6*72+5*9+4] + mr2.y * kernelsL[l][6*72+5*9+5] +
        bl2.y * kernelsL[l][6*72+5*9+6] + bc2.y * kernelsL[l][6*72+5*9+7] + br2.y * kernelsL[l][6*72+5*9+8] + 
)"
R"(
        tl2.z * kernelsL[l][6*72+6*9+0] + tc2.z * kernelsL[l][6*72+6*9+1] + tr2.z * kernelsL[l][6*72+6*9+2] +
        ml2.z * kernelsL[l][6*72+6*9+3] + mc2.z * kernelsL[l][6*72+6*9+4] + mr2.z * kernelsL[l][6*72+6*9+5] +
        bl2.z * kernelsL[l][6*72+6*9+6] + bc2.z * kernelsL[l][6*72+6*9+7] + br2.z * kernelsL[l][6*72+6*9+8] + 

        tl2.w * kernelsL[l][6*72+7*9+0] + tc2.w * kernelsL[l][6*72+7*9+1] + tr2.w * kernelsL[l][6*72+7*9+2] +
        ml2.w * kernelsL[l][6*72+7*9+3] + mc2.w * kernelsL[l][6*72+7*9+4] + mr2.w * kernelsL[l][6*72+7*9+5] +
        bl2.w * kernelsL[l][6*72+7*9+6] + bc2.w * kernelsL[l][6*72+7*9+7] + br2.w * kernelsL[l][6*72+7*9+8] + biasL[l][6]
        ,
        tl1.x * kernelsL[l][7*72+0*9+0] + tc1.x * kernelsL[l][7*72+0*9+1] + tr1.x * kernelsL[l][7*72+0*9+2] +
        ml1.x * kernelsL[l][7*72+0*9+3] + mc1.x * kernelsL[l][7*72+0*9+4] + mr1.x * kernelsL[l][7*72+0*9+5] +
        bl1.x * kernelsL[l][7*72+0*9+6] + bc1.x * kernelsL[l][7*72+0*9+7] + br1.x * kernelsL[l][7*72+0*9+8] + 

        tl1.y * kernelsL[l][7*72+1*9+0] + tc1.y * kernelsL[l][7*72+1*9+1] + tr1.y * kernelsL[l][7*72+1*9+2] +
        ml1.y * kernelsL[l][7*72+1*9+3] + mc1.y * kernelsL[l][7*72+1*9+4] + mr1.y * kernelsL[l][7*72+1*9+5] +
        bl1.y * kernelsL[l][7*72+1*9+6] + bc1.y * kernelsL[l][7*72+1*9+7] + br1.y * kernelsL[l][7*72+1*9+8] + 

        tl1.z * kernelsL[l][7*72+2*9+0] + tc1.z * kernelsL[l][7*72+2*9+1] + tr1.z * kernelsL[l][7*72+2*9+2] +
        ml1.z * kernelsL[l][7*72+2*9+3] + mc1.z * kernelsL[l][7*72+2*9+4] + mr1.z * kernelsL[l][7*72+2*9+5] +
        bl1.z * kernelsL[l][7*72+2*9+6] + bc1.z * kernelsL[l][7*72+2*9+7] + br1.z * kernelsL[l][7*72+2*9+8] + 

        tl1.w * kernelsL[l][7*72+3*9+0] + tc1.w * kernelsL[l][7*72+3*9+1] + tr1.w * kernelsL[l][7*72+3*9+2] +
        ml1.w * kernelsL[l][7*72+3*9+3] + mc1.w * kernelsL[l][7*72+3*9+4] + mr1.w * kernelsL[l][7*72+3*9+5] +
        bl1.w * kernelsL[l][7*72+3*9+6] + bc1.w * kernelsL[l][7*72+3*9+7] + br1.w * kernelsL[l][7*72+3*9+8] +

        tl2.x * kernelsL[l][7*72+4*9+0] + tc2.x * kernelsL[l][7*72+4*9+1] + tr2.x * kernelsL[l][7*72+4*9+2] +
        ml2.x * kernelsL[l][7*72+4*9+3] + mc2.x * kernelsL[l][7*72+4*9+4] + mr2.x * kernelsL[l][7*72+4*9+5] +
        bl2.x * kernelsL[l][7*72+4*9+6] + bc2.x * kernelsL[l][7*72+4*9+7] + br2.x * kernelsL[l][7*72+4*9+8] + 

        tl2.y * kernelsL[l][7*72+5*9+0] + tc2.y * kernelsL[l][7*72+5*9+1] + tr2.y * kernelsL[l][7*72+5*9+2] +
        ml2.y * kernelsL[l][7*72+5*9+3] + mc2.y * kernelsL[l][7*72+5*9+4] + mr2.y * kernelsL[l][7*72+5*9+5] +
        bl2.y * kernelsL[l][7*72+5*9+6] + bc2.y * kernelsL[l][7*72+5*9+7] + br2.y * kernelsL[l][7*72+5*9+8] + 

        tl2.z * kernelsL[l][7*72+6*9+0] + tc2.z * kernelsL[l][7*72+6*9+1] + tr2.z * kernelsL[l][7*72+6*9+2] +
        ml2.z * kernelsL[l][7*72+6*9+3] + mc2.z * kernelsL[l][7*72+6*9+4] + mr2.z * kernelsL[l][7*72+6*9+5] +
        bl2.z * kernelsL[l][7*72+6*9+6] + bc2.z * kernelsL[l][7*72+6*9+7] + br2.z * kernelsL[l][7*72+6*9+8] + 

        tl2.w * kernelsL[l][7*72+7*9+0] + tc2.w * kernelsL[l][7*72+7*9+1] + tr2.w * kernelsL[l][7*72+7*9+2] +
        ml2.w * kernelsL[l][7*72+7*9+3] + mc2.w * kernelsL[l][7*72+7*9+4] + mr2.w * kernelsL[l][7*72+7*9+5] +
        bl2.w * kernelsL[l][7*72+7*9+6] + bc2.w * kernelsL[l][7*72+7*9+7] + br2.w * kernelsL[l][7*72+7*9+8] + biasL[l][7]
    ));

    write_imagef(tmpImgOut, coord1, c1234);
    write_imagef(tmpImgOut, coord2, c5678);
}

__kernel void convTranspose8To1(
    __read_only image2d_array_t tmpImgIn,
    __write_only image2d_t dstImg)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(dstImg) || y >= get_image_height(dstImg))
        return;

    int2 coord = (int2)(x, y);
    int2 orgCoord = coord / 2;

    int2 pos = coord & 1;
    int index = pos.y * 2 + pos.x;

    float4 mc1 = read_imagef(tmpImgIn, samplerN, (int4)(orgCoord, 0, 0));
    float4 mc2 = read_imagef(tmpImgIn, samplerN, (int4)(orgCoord, 1, 0));

    float c = clamp(
        mc1.x * kernelsL10[0 + index] +
        mc1.y * kernelsL10[4 + index] +
        mc1.z * kernelsL10[8 + index] +
        mc1.w * kernelsL10[12 + index] +
        mc2.x * kernelsL10[16 + index] +
        mc2.y * kernelsL10[20 + index] +
        mc2.z * kernelsL10[24 + index] +
        mc2.w * kernelsL10[28 + index], 0.0f, 1.0f);

    write_imagef(dstImg, coord, (float4)(c, 0.0f, 0.0f, 1.0f));
})";

const std::string Anime4KCPP::OpenCL::ACNet::ACNetKernelSourceString[TotalTypeCount] =
{
std::string(
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
 0.0609,  0.1027, -0.0447,
-0.1423,  0.7196,  0.1803,
 0.0842,  0.0696,  0.0082,
 0.0089,  0.1540, -0.8589,
 0.0448,  0.8659, -0.2420,
-0.0364,  0.0585,  0.0125,
-0.1937,  0.7259,  0.0119,
-0.8266,  0.4147,  0.0088,
-0.0453, -0.0451, -0.0182,
 0.0264, -0.9422,  0.1258,
-0.0543,  0.1282,  0.7102,
-0.0106,  0.0386, -0.0141,
 0.2054, -0.0393,  0.1494,
 0.3106,  0.5722,  0.2640,
 0.1708, -0.1640, -0.0212,
 0.0558, -0.2887, -0.1666,
 0.3123, -0.3097, -0.2281,
 0.2880,  0.3001,  0.0526,
-0.0320,  0.0584, -0.0193,
-0.0135,  1.0649, -0.1246,
 0.0283, -0.3030, -0.6378,
-0.0040, -0.9122,  0.0181,
 0.0365,  0.8947, -0.0420,
-0.0199,  0.0217,  0.0060
};

__constant float biasL1[8] = 
{
-0.7577, -0.0210,  0.0292, -0.0189,  0.0223,  0.0340,  0.0150, -0.0044
};
)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
 2.0611e-01,  6.6865e-02, -9.9123e-02,
 8.5279e-02, -4.5549e-02, -2.9491e-02,
-1.0358e-01, -2.4844e-02, -8.1539e-03,
-1.1308e-01, -6.4228e-02, -8.8081e-02,
 2.7810e-02, -1.6054e-01, -1.1985e-01,
-2.8679e-01, -1.7785e-02,  1.1559e-01,
 2.1614e-02, -6.8870e-02, -2.4707e-01,
 9.6867e-02, -1.6561e-01,  2.8281e-02,
-8.2469e-02, -9.8554e-02, -1.7147e-02,
 3.3710e-01,  9.2126e-02,  3.6880e-02,
 5.7004e-02,  4.0175e-02,  1.6116e-01,
 2.5629e-01,  5.1154e-01,  2.4119e-02,
 1.9495e-02,  2.6940e-01, -1.4050e-01,
 5.0325e-02, -4.5920e-02, -1.3586e-01,
 5.9458e-02,  1.3860e-01, -2.1065e-01,
-1.0744e-01, -1.5915e-01, -1.1528e-02,
-1.1470e-01,  6.3455e-02, -5.5558e-02,
-6.9920e-02, -3.0142e-02, -4.9059e-02,
 3.6421e-01,  3.0252e-01, -1.3562e-01,
 1.5238e-01, -1.9868e-01, -3.2644e-02,
-4.2849e-02,  1.3677e-02,  7.3854e-02,
 7.6609e-02, -1.0121e-01,  3.6319e-02,
 9.3536e-02,  6.0386e-02,  1.0086e-01,
-2.6630e-01,  2.5875e-02, -1.9225e-01,
 4.0687e-02,  1.1005e-01,  9.9578e-03,
 1.6939e-01,  5.0872e-01,  8.9876e-02,
 6.9561e-02,  1.1910e-01, -1.8091e-02,
-3.5739e-02, -7.5300e-02, -1.6788e-02,
 3.0316e-02,  1.5942e-01, -9.0878e-02,
-6.3737e-02,  2.6141e-02,  8.8040e-03,
 3.4954e-03, -6.6707e-02,  1.4551e-01,
 7.6258e-02,  1.4893e-01, -1.5255e-01,
 6.2442e-02,  2.2166e-01,  7.5327e-02,
 5.4785e-02, -1.4503e-02, -1.5188e-03,
 1.6748e-01, -5.2731e-03, -1.9900e-02,
 4.4786e-02, -1.0669e-01,  1.3192e-01,
 1.9961e-02, -8.1015e-02, -3.2264e-02,
 1.0544e-01,  1.8844e-01,  7.4274e-03,
 6.6729e-02, -7.8318e-02,  3.0775e-02,
-8.6109e-03,  7.4977e-02,  9.4079e-02,
-1.2726e-01, -2.9664e-01,  7.8153e-03,
-4.8413e-02, -1.8450e-01, -7.1065e-02,
-8.7609e-02, -7.7192e-02,  5.0919e-02,
-1.4021e-01,  3.5696e-01,  1.2079e-02,
-2.0318e-02, -1.8827e-02,  3.9084e-02,
-2.8654e-02, -6.4166e-02,  5.4889e-02,
 8.2689e-02,  8.4463e-02,  2.2339e-02,
 1.0805e-01, -1.2566e-01,  1.7109e-01,
-6.1338e-02, -3.4043e-02,  4.0473e-02,
 6.3821e-02,  1.7626e-01, -5.8112e-02,
-9.5002e-02,  1.3327e-02,  1.2242e-01,
 4.9008e-02, -4.3678e-02,  2.2362e-02,
-7.7903e-02, -3.8252e-02, -5.2271e-02,
-1.8884e-02, -1.2859e-01,  4.1172e-02,
-3.1181e-02,  3.2348e-02, -4.9081e-02,
-6.7966e-02, -2.4896e-02, -6.5323e-02,
 8.0742e-02,  2.6093e-01, -2.4638e-01,
-8.0881e-02, -2.9643e-02, -7.9627e-02,
 1.4020e-01,  2.1575e-01,  8.1244e-03,
 2.1561e-01, -2.9305e-01, -2.5535e-02,
-8.5538e-02, -1.4456e-01, -7.5664e-02,
-3.9921e-02,  4.0659e-02,  1.7812e-01,
 1.1580e-01,  5.6628e-02,  9.0008e-02,
-2.2384e-02, -1.9788e-02, -4.0547e-02,
 1.0070e-01,  2.9581e-01,  1.9936e-01,
-1.1957e-01, -8.6508e-02, -8.2543e-04,
-5.2879e-02,  1.5486e-01,  1.0829e-02,
 1.4716e-01,  3.4257e-01, -3.2058e-03,
-2.1687e-02,  5.8641e-02, -6.3806e-02,
-3.2607e-02,  7.3328e-02, -6.4738e-03,
-1.0031e-01, -1.7698e-01, -9.4201e-02,
-3.3644e-02, -3.5860e-01, -9.3200e-02,
-7.4142e-02,  5.5001e-02,  4.3741e-02,
-2.2447e-03,  1.1941e-01, -1.6135e-02,
-1.4764e-02, -1.0194e-02,  3.2540e-02,
-1.0588e-01, -2.3000e-01, -1.1557e-02,
-9.0254e-02,  2.3352e-01, -1.3622e-01,
-1.9256e-03, -5.3372e-02,  1.0314e-01,
-2.0100e-02,  1.0700e-01,  1.6108e-01,
 2.8422e-02,  2.7909e-01,  3.8342e-01,
 1.4025e-02,  9.0965e-02,  2.0218e-01,
 3.3562e-03,  7.6652e-02,  4.5974e-02,
-1.3617e-02, -1.4014e-01, -1.9253e-02,
 1.1020e-01, -1.9678e-01,  6.7123e-02,
-3.3294e-02, -1.3006e-01, -1.0111e-01,
 5.5813e-02,  2.1127e-01,  2.0248e-02,
-9.6386e-04, -5.2497e-03,  1.1134e-01,
 2.8910e-02,  1.2229e-01,  1.8439e-01,
 1.6413e-02,  1.5870e-01, -1.1616e-01,
-1.6032e-03, -6.8258e-03, -2.1883e-02,
 1.2052e-01, -2.1982e-02, -1.3088e-01,
 2.8664e-02, -5.0670e-02,  2.2927e-01,
 2.0461e-02,  7.7250e-03, -2.6630e-02,
-9.0406e-02, -1.4174e-01,  9.8969e-02,
-6.6573e-02, -2.4425e-01, -3.5126e-02,
 9.3859e-02,  1.9058e-01, -1.6569e-01,
-4.9163e-03,  7.4149e-02,  6.3345e-02,
-1.7888e-02, -9.1876e-02,  1.3728e-01,
-9.6098e-02, -3.4814e-02, -1.0862e-02,
 4.8031e-03,  2.5206e-01,  8.0316e-02,
 1.5102e-01,  4.1236e-02,  2.2339e-01,
 2.8500e-01,  1.5106e-01,  9.6321e-04,
-6.0741e-02,  3.5759e-02, -1.8829e-01,
-1.1295e-03, -6.2322e-02,  8.4974e-01,
-3.9817e-02, -2.0666e-01,  2.2961e-01,
 3.6857e-02, -2.0211e-02, -9.3342e-02,
 2.0827e-02,  6.8874e-02, -6.0287e-02,
-6.9724e-02,  1.4423e-01, -7.6017e-02,
 1.4718e-02,  1.8990e-01,  1.1789e-01,
-1.5018e-01, -2.3071e-01,  1.7511e-01,
-7.7605e-02,  5.0621e-02, -1.0381e-01,
 8.6845e-02, -1.2410e-01, -4.4669e-01,
 2.7930e-02, -5.4713e-02, -7.7923e-02,
 8.6000e-02, -2.6371e-02, -8.6541e-02,
-1.1521e-01,  1.4389e-01,  5.0507e-02,
-1.6618e-02, -2.5150e-01, -4.9759e-02,
 7.7166e-02,  4.5033e-03, -5.4649e-02,
 2.8548e-03, -2.8078e-03,  8.1129e-02,
-4.5973e-02,  3.6740e-03,  2.0746e-01,
-9.8191e-02,  1.2807e-01,  8.1950e-03,
 1.4240e-01,  1.5104e-01,  6.9624e-02,
 2.2309e-01,  2.5688e-01,  9.4766e-02,
 6.2560e-02,  7.1347e-02,  4.1432e-02,
-3.1829e-02,  1.5207e-01,  2.0575e-02,
-1.2506e-01,  2.9274e-01,  9.4712e-02,
-2.0520e-01,  4.9894e-04,  5.6171e-02,
-4.1567e-03,  6.6753e-02, -1.5767e-01,
 6.3768e-02,  8.3008e-02, -3.5639e-01,
 4.4660e-02,  2.6996e-01, -6.4014e-02,
 8.5475e-02,  1.7854e-02, -6.4079e-02,
 1.8760e-01,  1.5285e-01, -3.5614e-02,
 1.0747e-02, -3.1330e-01, -4.8664e-02,
 7.2150e-02,  1.7570e-01,  1.6716e-01,
 6.2431e-02,  2.3755e-01,  2.8554e-01,
 3.5791e-02,  2.8185e-01,  1.5810e-01,
-4.0886e-02,  1.8833e-02, -8.2903e-03,
 1.3994e-02, -1.0846e-01,  3.5315e-02,
-6.2674e-02,  6.2806e-02,  2.2168e-02,
-3.6236e-01, -2.5326e-01,  5.6331e-02,
 9.8762e-02,  3.8049e-01,  5.9885e-02,
-3.0541e-02,  7.9855e-02, -5.8639e-02,
 1.1104e-03,  1.7147e-02,  3.3115e-02,
-3.3663e-02,  7.4615e-02,  6.4211e-02,
-7.3441e-02, -1.5568e-01,  7.6546e-02,
 6.1802e-02, -1.5300e-01, -1.8209e-02,
-9.2786e-03,  1.6622e-01,  1.1354e-01,
 9.5865e-03, -2.4226e-02, -1.4750e-03,
-5.5294e-02, -1.1839e-01,  3.8867e-03,
 1.7262e-01,  4.2743e-01,  6.8970e-02,
-2.0232e-01, -1.4564e-01,  2.3025e-02,
-2.6139e-03, -1.6907e-02,  1.1693e-01,
-9.4871e-03,  3.8488e-02, -4.8351e-02,
-9.2171e-02,  4.8227e-02,  9.7378e-02,
-1.0292e-01, -1.2084e-01, -9.6676e-02,
 1.8103e-02,  3.0658e-01, -7.7755e-02,
-2.4362e-02, -1.9862e-01, -6.9665e-02,
 8.2944e-03, -1.4680e-01, -1.7371e-02,
-1.6534e-01,  2.5752e-01,  1.1129e-01,
-9.4151e-02, -1.3225e-01,  1.5933e-01,
 9.0723e-02,  5.5469e-02, -1.4091e-01,
 8.3404e-02,  1.3741e-01, -3.5438e-02,
 3.2681e-02,  2.8491e-02,  1.4278e-02,
 2.3789e-01, -2.3687e-03, -5.3264e-03,
-1.1161e-01,  1.9351e-02,  5.0832e-02,
 8.2246e-03,  2.9892e-02, -3.7197e-02,
 4.8236e-02,  1.6945e-01,  1.3673e-01,
 1.1236e-01,  7.2318e-01, -4.1618e-02,
 2.7494e-01,  1.0081e-01, -8.5399e-03,
-5.6151e-02,  8.1212e-02, -7.5770e-02,
 2.7872e-02,  9.4644e-02,  1.1175e-02,
-6.1539e-02,  7.7395e-02, -3.2495e-02,
-5.1640e-02,  2.1028e-03,  1.5825e-02,
-1.1004e-01,  2.3153e-01, -6.1653e-02,
-2.6497e-02,  5.9461e-01,  4.0865e-02,
-1.9956e-02,  7.9328e-02, -1.7002e-02,
-5.5930e-03,  5.2015e-02,  7.7945e-04,
 1.0136e-02, -9.0111e-02, -1.1175e-01,
-3.1781e-02,  1.4686e-01, -7.5718e-03,
 1.1036e-02,  2.4618e-01,  8.5951e-02,
 3.4775e-02, -1.2184e-01,  1.8010e-01,
-3.6781e-02, -1.3912e-01, -4.9172e-02,
 3.3064e-02,  5.0582e-01,  1.0713e-02,
-1.2934e-02, -1.7697e-01, -1.4954e-01,
 2.2229e-02, -5.8568e-03, -5.0186e-02,
 1.9648e-02, -1.1302e-01,  1.5629e-02,
-3.5015e-02,  9.5032e-02, -2.9677e-02,
 9.5173e-02, -3.0330e-02, -3.7652e-02,
-2.6097e-03,  7.4723e-01, -7.6234e-03,
-3.8826e-02,  1.0191e-01,  3.6589e-03,
-2.6503e-02, -1.1133e-01, -2.2029e-02,
-1.9101e-01, -2.1108e-01, -7.4371e-02,
-7.9349e-02, -1.0405e-01,  5.0315e-02
}
,)"
R"(
{
-4.2606e-02, -8.9001e-02, -6.4006e-02,
 1.1132e-01,  7.6609e-02,  8.6417e-02,
 7.6477e-03, -1.6416e-02, -8.2094e-02,
 1.0779e-01,  2.1837e-01,  1.8094e-01,
-2.6306e-02, -1.2452e-01,  1.2662e-02,
 3.1633e-02,  1.8717e-02,  3.1043e-02,
 4.0927e-02,  5.0311e-02,  1.1648e-01,
 2.2429e-01,  2.0757e-01,  4.3662e-03,
 3.6341e-02, -4.7637e-02,  8.3645e-02,
-8.9260e-03,  1.8507e-02,  7.9069e-02,
-1.9411e-01, -8.6847e-02, -3.6639e-03,
 4.0328e-02, -3.6821e-02, -8.5387e-02,
 5.8173e-02,  5.9991e-02, -3.1398e-02,
 1.5818e-01,  3.0861e-01, -2.3818e-02,
 1.2176e-01,  6.7520e-02,  8.9401e-02,
-2.8859e-02, -1.2237e-01, -1.0625e-01,
 3.1675e-02,  1.4172e-01, -1.4373e-01,
 1.4653e-02,  1.0205e-01,  6.2557e-02,
-8.7292e-02, -2.1255e-02,  3.6830e-02,
-5.4417e-02,  3.0501e-01,  1.6897e-01,
-2.2187e-02, -8.9609e-02, -2.2830e-02,
 4.9846e-02,  3.3395e-01, -3.1561e-02,
-1.3191e-02,  4.2663e-01, -6.9727e-02,
 1.4570e-02, -4.0002e-02,  5.6394e-02,
-8.2547e-02,  1.9249e-01,  1.5591e-01,
 1.4536e-01, -1.0409e-01,  1.2382e-01,
 1.8189e-01,  9.2917e-02, -1.4394e-01,
-5.6260e-02, -2.7043e-01,  1.5392e-02,
-1.4305e-02,  1.1131e-01, -8.5913e-02,
 7.7914e-02, -6.5484e-03, -1.8375e-01,
-1.4059e-01, -5.7339e-01, -3.9073e-02,
-1.1701e-01, -3.1806e-02,  7.7726e-02,
 2.1688e-02,  9.9297e-02,  3.8224e-02,
 7.9884e-02,  5.2461e-02,  1.0318e-01,
 4.0054e-02,  1.4695e-01,  1.2577e-01,
-1.8790e-03, -4.9421e-02,  2.3235e-02,
-8.9820e-02, -1.6994e-01, -1.5986e-01,
 2.3436e-01, -1.5346e-01,  1.5014e-02,
-3.9139e-02, -7.9388e-02, -4.9057e-02,
-1.1193e-01, -2.5705e-01,  1.1995e-01,
 5.7929e-02,  2.4988e-01, -4.9406e-03,
-3.9363e-02, -1.1691e-02, -1.2236e-03,
-2.0521e-01,  2.1901e-01,  1.5957e-01,
 2.1062e-01, -1.4157e-01, -3.4340e-01,
 3.8520e-02, -2.0820e-01,  2.4570e-03,
 1.7211e-01,  2.0214e-01,  1.3821e-01,
-7.1520e-02,  1.4847e-01, -1.3820e-01,
-2.4712e-02, -1.5925e-02,  1.7403e-02,
-3.7515e-02,  3.0461e-02, -2.7543e-02,
 8.6148e-02, -6.1486e-02,  1.2610e-02,
 2.9748e-03,  1.1778e-01,  2.9032e-02,
-2.1706e-02, -2.2406e-02,  2.6769e-02,
-3.6965e-02,  2.2180e-01, -4.0929e-02,
-3.2629e-03,  8.3419e-02, -1.4587e-01,
-1.3909e-02, -2.0166e-02, -1.0029e-01,
 7.6360e-02,  8.0819e-02, -1.0933e-01,
-5.8919e-02,  2.4745e-02,  3.7375e-02,
-1.1333e-02,  1.4747e-02, -7.8958e-02,
-3.1535e-02,  1.7403e-01,  1.3946e-02,
-3.2038e-02,  5.1151e-02, -6.1063e-02,
-8.6472e-03, -6.9689e-02,  5.6846e-03,
 5.7914e-02, -1.9818e-01, -7.5321e-02,
 8.7453e-02,  7.8354e-02,  2.1997e-02,
-4.7606e-02,  1.3915e-01,  1.1653e-01,
 9.6050e-02,  4.0099e-01,  1.5631e-01,
 3.1492e-02,  2.4797e-01,  6.8716e-02,
-6.2664e-03,  9.1754e-02, -5.7244e-03,
 1.3538e-01,  1.5366e-01,  9.4916e-02,
-4.2115e-02, -3.6585e-01, -1.4559e-01,
 9.1550e-02, -5.4007e-02,  6.7482e-02,
-1.8687e-01,  3.2120e-01,  5.1031e-03,
-6.1205e-02, -5.1780e-02,  1.6442e-02,
-1.2316e-02, -1.3907e-01, -1.4446e-01,
-2.7899e-01, -8.5969e-02, -1.0870e-01,
-2.6157e-01,  8.9532e-02,  3.0958e-02,
-1.5393e-01, -4.2781e-02, -2.0951e-01,
 2.0328e-01,  4.5317e-01, -3.0467e-02,
-6.1346e-02,  1.0381e-01, -1.3719e-01,
-9.8572e-02, -1.4035e-01, -1.9431e-02,
 2.5542e-02,  3.2609e-01,  1.7983e-03,
-1.0800e-01, -2.9022e-02,  6.2691e-03,
 2.8937e-02, -1.3483e-01, -4.1655e-02,
 2.0172e-01,  1.4283e-02,  9.6200e-02,
 1.9027e-02,  3.1240e-01, -2.9553e-02,
 6.2776e-02,  1.3845e-01,  4.5834e-02,
-2.3854e-01, -4.0267e-02,  1.5634e-02,
-1.9246e-01, -3.2332e-02,  3.2442e-03,
-6.1880e-02, -8.8192e-02, -6.0172e-02,
 2.5002e-01,  1.5148e-01,  6.4459e-02,
-2.1022e-01, -8.3893e-02,  6.9554e-03,
 7.0244e-02, -2.9551e-02,  1.6481e-02,
-3.1036e-02, -2.0026e-01, -8.4748e-02,
-1.3108e-01, -1.3784e-01,  9.4900e-02,
-2.1256e-01, -4.1767e-02,  8.4665e-02,
-4.0235e-01,  1.0604e-01, -3.1827e-02,
-4.9825e-02, -9.1267e-04,  1.5527e-02,
-6.5729e-03, -1.8932e-02, -3.4591e-02,
 1.1066e-01,  9.3979e-02,  2.6059e-02,
-1.2395e-01, -2.4768e-01, -1.6304e-01,
 8.8329e-03, -2.1606e-02, -4.0878e-02,
-1.5581e-02, -1.4829e-02, -1.5959e-02,
-1.0463e-04, -4.2903e-03, -4.6657e-02,
 2.2995e-02,  1.7917e-02, -9.1404e-02,
-1.2326e-01,  1.4582e-01, -7.0959e-02,
-1.8058e-02, -8.5228e-02,  4.2799e-02,
-2.2829e-03,  8.6577e-02, -1.1909e-01,
-1.8061e-01,  1.1166e-01, -8.2255e-02,
-1.3190e-01,  7.7123e-02,  2.3224e-02,
 1.8661e-02,  2.4461e-02,  3.6060e-02,
-4.5224e-02, -1.7672e-01,  1.6080e-01,
-4.2175e-01, -2.2557e-01, -1.0719e-01,
-2.9506e-02,  9.5020e-02, -6.6465e-02,
-7.2627e-02,  3.1236e-01,  5.5764e-02,
-2.8789e-01, -1.8915e-01,  9.0825e-02,
-5.8618e-02,  6.4082e-02,  4.8461e-03,
-5.9405e-02,  3.2644e-01, -7.1278e-02,
-1.8084e-01,  2.0858e-02, -9.3690e-03,
-7.6565e-03, -9.6854e-02,  7.6121e-03,
 1.4791e-01,  4.5612e-01,  1.9889e-02,
-5.5498e-02, -1.1266e-01,  2.2790e-02,
-3.8821e-02, -1.5780e-02,  1.2549e-02,
-3.8232e-02, -2.8870e-01,  2.6216e-02,
 1.0375e-01, -2.9621e-02,  1.8479e-03,
 5.0207e-02,  1.5189e-01,  1.2533e-01,
 1.8298e-01, -1.2870e-01,  3.0681e-01,
-1.9571e-02, -8.6302e-02,  9.1121e-02,
 1.0113e-01, -1.8362e-01,  3.2642e-02,
 1.7034e-01, -3.1077e-01, -4.8737e-02,
 5.9144e-02,  5.6052e-03,  3.2360e-02,
-9.0123e-02,  7.7996e-02,  3.6297e-02,
-3.4389e-01,  1.1841e-01, -2.0900e-02,
 9.4930e-02, -9.1504e-02, -4.5308e-02,
 3.7723e-03, -3.7580e-02, -6.6410e-02,
 5.2501e-02, -1.2530e-01,  3.5944e-02,
 3.8378e-02,  9.5188e-02,  2.1952e-03,
-2.4333e-02,  2.7977e-01,  5.6961e-02,
-3.0605e-03,  8.3684e-02,  4.4848e-03,
-7.8935e-02, -1.9544e-01, -5.3311e-02,
-2.6595e-02,  1.2278e-01, -3.1659e-02,
-1.0103e-02,  4.7763e-01,  2.5359e-02,
 8.1397e-02,  3.0548e-01,  9.7097e-02,
 3.6232e-02, -1.1091e-01,  1.2841e-01,
 1.9277e-01,  2.9322e-01, -1.6740e-01,
 1.2107e-01, -6.2883e-02,  4.0603e-02,
-1.5750e-01, -8.6183e-02, -1.4194e-01,
 1.1932e-01, -3.9175e-01, -5.4495e-02,
-1.4001e-02, -2.0594e-01, -8.2683e-02,
 8.6156e-02,  2.1499e-02,  2.2080e-01,
 5.5703e-02, -3.6307e-01,  8.3129e-02,
 8.9280e-02, -3.5897e-02,  1.6106e-01,
 9.1171e-02, -3.1102e-01,  1.2425e-01,
 1.0278e-01, -3.1014e-01, -6.9138e-02,
 8.0839e-02, -3.6183e-02,  1.0341e-01,
-1.8334e-01, -5.3700e-02,  2.3336e-01,
-1.4464e-01, -5.0320e-01, -2.9836e-02,
-1.7225e-01, -3.9499e-01, -1.7321e-01,
 1.7510e-01,  1.7897e-01, -2.6518e-01,
 2.3638e-01,  5.0270e-01, -4.9731e-03,
 2.2603e-01,  2.5317e-01,  2.4079e-01,
-1.3159e-01,  1.5638e-01,  1.2480e-01,
-6.2164e-02,  7.9458e-02, -9.4804e-02,
 8.5690e-03,  7.4971e-03,  8.6630e-02,
-1.3148e-02,  6.8660e-02, -7.4230e-03,
 2.9702e-02,  1.2036e-01,  9.5504e-02,
-3.2694e-03,  8.6722e-02, -6.2433e-02,
 3.2527e-01,  3.2087e-01, -9.4429e-05,
 1.3556e-01, -7.0413e-02,  2.9383e-02,
 2.0617e-02,  3.3218e-02,  4.4898e-02,
-4.8260e-01, -2.1329e-01,  1.5890e-02,
-2.6600e-01, -8.8519e-02, -4.3800e-02,
-1.7299e-01, -2.0757e-01, -2.6658e-01,
 6.9707e-02, -4.4700e-02,  6.5570e-02,
 2.3992e-01,  1.5078e-01,  2.8713e-02,
-9.1197e-02,  1.9765e-02, -1.8751e-02,
-9.9277e-02, -3.1437e-01,  4.0730e-02,
 2.4208e-02, -8.8322e-02, -1.6245e-01,
 1.3037e-02, -3.4708e-02, -4.4285e-02,
-1.3592e-01, -1.3575e-01, -7.4546e-02,
 1.4670e-01, -1.3366e-01,  2.1553e-03,
 8.1235e-03, -1.2068e-01, -5.7287e-02,
 1.8015e-01,  2.1390e-01,  8.6923e-03,
 2.8833e-01,  6.6345e-02,  1.4578e-01,
 2.2338e-01,  2.6453e-01, -2.9112e-02,
 1.4018e-01, -9.2824e-02, -2.2795e-02,
 1.2360e-01,  2.2527e-01, -1.1817e-01,
-3.8872e-02, -1.9982e-02, -7.7514e-02,
 1.7744e-03,  3.1736e-02,  4.5882e-02,
-2.5222e-02,  2.4298e-01, -3.8596e-02,
 1.2545e-02,  3.1872e-02,  7.1925e-02,
 7.9782e-02, -1.5533e-01, -1.4619e-02,
-1.2223e-01, -1.8631e-03, -9.8832e-02,
-1.6815e-02, -8.1440e-02,  6.8038e-02
}
,)"
R"(
{
 2.3898e-02,  1.2411e-02, -3.2770e-02,
-2.6029e-01,  3.2690e-01, -1.8246e-01,
 1.1224e-02,  8.0193e-02, -5.0412e-02,
-9.3849e-02,  2.0325e-02,  2.6309e-02,
 1.2266e-02,  1.7698e-01,  2.7049e-01,
 1.2918e-01,  2.0190e-01,  2.7352e-01,
-7.2100e-02,  1.3357e-01, -1.3702e-01,
 2.2527e-01,  1.5821e-01, -2.3104e-01,
 1.0182e-02, -1.5499e-01,  7.1906e-02,
 1.5865e-01,  7.0950e-02, -6.3336e-02,
 2.2661e-01, -4.2997e-01, -4.2013e-01,
 1.7549e-02, -1.3142e-01, -3.1663e-01,
 1.3617e-01,  1.4229e-01, -1.0707e-02,
-1.0986e-02,  2.8816e-01, -3.6239e-01,
 2.2579e-02, -1.4332e-02,  7.1339e-03,
-1.4357e-01, -9.7608e-02,  1.4646e-01,
-5.3856e-02,  3.3898e-01, -2.4936e-01,
-2.9500e-02,  2.1799e-02,  1.1901e-02,
 3.6996e-02,  2.1291e-02,  3.2150e-02,
 9.8375e-02,  2.4476e-01,  2.2896e-01,
 1.8392e-01, -7.4510e-02, -1.0152e-01,
 4.4757e-02, -4.8053e-03, -6.7254e-02,
-4.8370e-02, -7.8975e-02, -3.6007e-01,
-3.8160e-02,  8.7707e-02, -1.4986e-01,
-8.7544e-03, -4.3522e-02,  7.3822e-02,
-1.4523e-01,  1.1433e-01,  4.4109e-02,
-1.6025e-03,  2.5459e-02, -9.3562e-02,
-2.9192e-02, -1.0975e-01, -5.0943e-02,
-1.1215e-01,  1.9907e-01,  7.9934e-02,
 3.7066e-02,  3.0796e-01, -1.4034e-01,
-8.2315e-02, -2.0182e-02, -1.2824e-02,
-4.8007e-03,  1.2655e-01, -2.5157e-02,
 2.7796e-02, -4.3032e-02,  2.5397e-02,
 6.9377e-02,  2.3642e-01,  1.2713e-01,
 2.7878e-02, -1.5325e-01, -1.4871e-01,
 1.5800e-02, -4.5935e-02,  1.7370e-01,
 4.8058e-02, -1.8725e-01, -6.7048e-03,
-1.3932e-01, -6.0768e-02, -1.6976e-01,
-2.1189e-02,  1.0311e-02, -2.2970e-02,
-7.0546e-03,  7.9481e-02,  1.2146e-02,
 4.2666e-02,  3.5383e-01,  1.4381e-01,
 5.4384e-02, -9.3862e-02,  4.8870e-03,
 2.1141e-02, -6.6826e-02, -1.8526e-01,
 1.3309e-01,  3.3452e-01,  1.1058e-02,
-1.6967e-02,  1.1094e-01,  5.3230e-02,
 3.0409e-02, -4.7613e-02, -1.7737e-01,
-1.6678e-02, -7.8644e-02,  1.1743e-01,
 7.3322e-02, -1.1354e-01, -1.5737e-02,
-1.2397e-03, -1.4685e-02, -1.0192e-02,
 1.6045e-01,  3.6331e-02,  1.2219e-01,
 1.3123e-01,  5.7578e-02,  1.0291e-01,
 1.7424e-01,  1.0688e-01,  1.4263e-01,
 8.9942e-02, -2.7141e-02,  3.1238e-02,
-4.0240e-02, -1.0930e-01, -2.1276e-01,
 1.0357e-01,  5.7673e-02,  1.0356e-02,
-2.0864e-01, -1.9405e-01,  2.5094e-01,
-4.8277e-03, -1.3758e-01,  1.1562e-01,
-1.0358e-01,  2.0631e-01, -9.1445e-03,
-1.7602e-01,  1.0200e-01,  3.0032e-02,
-1.1495e-02, -4.5077e-02, -6.4748e-02,
-2.3072e-02, -3.2342e-02,  1.4503e-02,
-3.7052e-02, -1.2206e-01,  5.5395e-02,
 2.8331e-02, -4.2812e-03,  6.9807e-02,
 4.3593e-02, -6.7373e-03,  1.2760e-02,
 3.2896e-03, -2.4007e-01, -5.2920e-02,
 2.5193e-02, -2.1480e-01,  8.4654e-02,
 2.2642e-02,  8.2132e-02, -2.3864e-02,
-2.9726e-01,  8.0405e-02, -1.3190e-02,
-1.1310e-01, -4.4342e-01, -6.3536e-02,
-6.7090e-02,  1.1797e-01,  1.5315e-01,
 7.7829e-02, -1.4494e-01,  1.0233e-01,
 9.7059e-02,  1.2772e-01, -2.4394e-02,
-2.6179e-02,  2.6721e-02,  1.1707e-02,
-4.8024e-02, -2.3366e-01, -1.6978e-01,
-2.4402e-01, -2.8572e-01, -2.4053e-02,
-2.7451e-03,  7.1959e-02,  4.4706e-02,
-1.9900e-01,  2.1353e-01,  1.0625e-01,
 4.0246e-01,  4.2323e-01,  3.4046e-02,
-1.6943e-01, -2.0221e-01, -1.6369e-01,
 1.3882e-01,  2.1717e-01, -1.3581e-01,
 1.3975e-01,  1.1980e-01,  1.8888e-02,
-1.8110e-01, -2.6143e-01, -1.0109e-01,
 5.5844e-02, -1.2175e-01,  3.4447e-02,
 8.9688e-02,  2.4641e-01,  2.3287e-01,
-5.8259e-02, -1.3656e-01, -1.3936e-02,
-8.3429e-03,  2.3026e-01,  1.2302e-01,
-2.2969e-02,  6.0932e-02,  3.4749e-02,
 1.2910e-01,  2.4008e-01,  1.8908e-01,
-5.8776e-02,  3.8121e-01,  8.1312e-02,
 9.1175e-02, -1.8729e-02, -4.6156e-02,
 3.7493e-02, -3.5877e-02, -9.9651e-03,
 1.5864e-01,  1.3611e-01,  6.7880e-02,
 2.2216e-01,  9.3697e-02,  7.4782e-02,
-1.0861e-01, -2.5824e-01,  6.6455e-02,
 9.2238e-02, -2.3448e-01, -3.4057e-01,
-2.9658e-01,  9.4698e-03,  1.9315e-01,
-5.2396e-02,  1.2310e-01, -5.2917e-02,
-4.3708e-03,  1.9560e-01, -2.4309e-02,
-6.7388e-02, -8.8839e-02, -2.0907e-02,
 4.6550e-02,  3.4119e-02,  6.0977e-02,
-1.0054e-02,  1.4411e-01,  1.5622e-01,
 1.7401e-02,  2.5685e-01, -9.1853e-03,
-4.4530e-02, -1.8623e-01, -8.4557e-02,
 9.5962e-02,  2.6491e-01,  1.7854e-01,
-2.0547e-02, -1.2023e-01, -7.6897e-02,
-1.3418e-01, -1.4960e-01,  1.6292e-01,
-1.7275e-01, -6.0181e-02, -2.7034e-02,
-7.4189e-02, -3.5566e-02,  1.3995e-01,
 3.0758e-02,  3.3476e-02,  6.9837e-03,
-6.1089e-02, -9.6021e-02,  7.1716e-03,
 1.0389e-01,  4.7963e-02,  9.5921e-02,
 4.4569e-02,  1.2230e-01, -1.4417e-01,
-1.2825e-02,  3.1980e-01, -3.5905e-01,
-1.2557e-01, -7.5283e-02, -1.2343e-01,
 1.9791e-01,  7.9003e-02,  3.1163e-02,
 1.0969e-01,  1.6839e-01, -2.5816e-01,
-1.2617e-01,  1.3686e-01, -2.1078e-01,
-2.1870e-02, -1.8378e-01, -2.8893e-01,
-8.2523e-02, -3.0475e-02,  9.6007e-02,
 1.0669e-01, -1.4581e-03,  3.2441e-01,
-8.1872e-03,  1.1690e-02, -4.0179e-02,
-1.0835e-01,  3.6112e-01, -4.5990e-02,
-1.2355e-01, -1.3372e-01,  3.8136e-02,
-9.1530e-03,  3.5432e-02,  4.3950e-02,
-8.6859e-02,  1.5887e-01,  1.2796e-02,
 1.3554e-02, -1.5669e-01, -1.4371e-02,
-4.6609e-02,  1.7114e-01, -7.8284e-02,
 1.7611e-01,  4.1204e-01,  9.3281e-02,
 1.1420e-01,  1.2951e-01, -7.6025e-02,
-5.4831e-02,  9.7574e-02,  3.2839e-02,
 3.8475e-02, -6.0247e-02, -2.9627e-02,
-2.4367e-02,  1.3143e-02,  4.7017e-02,
 2.3800e-02, -2.4046e-02, -5.7044e-02,
 2.7280e-02,  7.8573e-01,  1.0079e-02,
 6.4100e-02,  5.1584e-02,  7.9653e-03,
-8.9480e-02, -1.6207e-01, -8.9418e-02,
-3.5589e-02,  3.5903e-01, -1.8381e-01,
 9.2356e-02,  8.8046e-02, -5.0229e-02,
 1.8609e-02,  1.1243e-01,  5.2599e-02,
-1.3374e-02, -3.3097e-01,  6.5346e-02,
 2.6760e-01, -1.0281e-01,  1.1607e-02,
 7.6576e-03, -3.5957e-02,  3.1924e-02,
-7.0088e-02,  9.1241e-02,  1.2827e-02,
 3.7165e-02,  7.0273e-03, -7.3945e-04,
-6.5406e-03,  7.2666e-02, -5.7348e-02,
-1.9100e-01, -7.4449e-02, -1.2496e-01,
 1.5299e-01, -8.8047e-02, -2.1810e-02,
-3.0241e-02, -7.4310e-03, -8.7682e-02,
-2.2479e-02,  9.6008e-02, -8.4539e-02,
-2.8915e-02,  1.7538e-01, -3.7735e-02,
-9.8463e-03, -6.9618e-02, -2.6095e-01,
 9.9950e-02,  5.0534e-01, -1.8812e-01,
-1.1986e-01,  7.1166e-02, -2.4769e-02,
 8.8529e-02,  9.8348e-02,  2.1136e-02,
-9.0337e-03,  1.3679e-01, -1.2115e-01,
-6.2478e-03,  1.1436e-01, -3.4610e-02,
-2.7350e-02,  1.0702e-01,  1.6220e-02,
 1.0912e-02,  1.0953e-01,  8.6762e-02,
 2.9348e-03, -2.2035e-02,  1.2376e-01,
 7.0102e-02, -1.0945e-01, -1.6640e-01,
-3.9916e-03, -2.6658e-02, -9.7031e-02,
-3.0047e-02,  1.6631e-03, -5.5031e-02,
-7.9624e-02,  1.9976e-01,  1.9582e-01,
 2.1377e-01,  3.5835e-01,  1.7012e-01,
-9.7751e-02,  4.9143e-01,  1.0988e-01,
 8.4055e-02, -7.3187e-03, -9.8808e-02,
 5.0590e-02, -8.9291e-02, -6.6857e-02,
 9.6737e-02, -3.0699e-01,  2.2889e-01,
 2.6727e-40, -5.2704e-40, -4.5038e-40,
-3.3108e-40,  5.2330e-40, -1.2724e-40,
-3.2957e-40, -5.8613e-40,  2.1618e-40,
-4.3882e-40, -3.3950e-40,  5.9372e-40,
 2.7277e-40, -1.3741e-40, -3.3597e-40,
 5.0687e-40,  4.7873e-40, -3.2116e-40,
-6.1388e-40, -6.0790e-40, -5.2667e-40,
-5.6524e-40, -6.1696e-40, -5.9796e-40,
 1.5824e-40, -5.2002e-40, -5.8960e-40,
-5.9860e-40,  3.6419e-40,  2.9975e-40,
-5.8988e-40,  3.3994e-40, -5.0611e-40,
 3.6410e-40,  2.9550e-40,  4.7468e-40,
 2.7503e-40, -3.4103e-40,  6.0339e-40,
-1.7691e-40,  6.7170e-41,  1.7101e-40,
 2.7166e-40,  4.3023e-40,  2.7735e-40,
-3.1937e-40, -4.9247e-40, -6.2495e-40,
 5.2938e-40, -3.3702e-40,  1.4976e-41,
 1.4031e-40, -4.6995e-40, -5.2409e-40,
 2.5460e-40,  2.6670e-40, -4.5339e-40,
 4.2896e-40, -5.7141e-40, -1.7003e-40,
 2.3597e-40,  1.3748e-40,  4.6163e-40,
 4.0680e-41, -6.1642e-40,  2.7304e-41,
 5.2250e-40, -3.9481e-40, -6.1808e-40,
 1.9462e-40,  2.6005e-40, -2.7281e-40
}
,)"
R"(
{
 1.3625e-02, -8.5594e-02, -1.9901e-01,
-6.4636e-02, -1.9030e-02,  4.1963e-02,
-7.5507e-02, -2.4474e-01, -4.2621e-02,
 2.8195e-02,  7.3102e-02, -9.3331e-02,
 7.7093e-02,  1.7800e-01, -7.6451e-02,
 2.8565e-02, -1.3540e-01, -1.9169e-01,
-1.8583e-02,  3.0135e-02,  8.1094e-03,
-1.2835e-01, -1.8041e-01, -8.9020e-02,
-8.2731e-02,  3.7861e-02, -9.4014e-02,
 4.6595e-02,  2.2052e-02, -1.5867e-01,
-1.0937e-02,  1.0030e-01, -1.3018e-01,
-9.1844e-02, -1.7508e-01,  2.2087e-01,
-9.3080e-02,  9.8069e-02, -7.0154e-02,
-6.6063e-02, -2.2142e-01,  4.1058e-01,
-6.5947e-02, -5.4662e-02,  9.9412e-02,
-5.1938e-02,  3.0932e-03,  1.8126e-01,
 3.6701e-02, -3.0349e-01,  9.9839e-02,
 2.5810e-02,  2.3644e-01, -2.4461e-01,
 2.1054e-01,  1.5630e-01, -1.9587e-01,
 5.0146e-02, -1.8844e-02,  3.6675e-01,
-4.0389e-03,  3.1596e-01,  3.6771e-03,
-2.2256e-40,  1.4272e-40, -2.0732e-40,
 5.5913e-40, -6.0538e-40,  1.2791e-40,
 4.5825e-41,  4.1080e-41, -1.8211e-40,
 2.2687e-01, -5.8992e-02,  4.7796e-03,
 6.0603e-01,  2.7961e-01,  1.5973e-02,
 2.3035e-01,  1.3031e-01, -9.9280e-03,
-4.7235e-02,  5.1773e-02, -4.8586e-02,
-1.4510e-01, -1.7336e-01,  1.0981e-01,
-2.0303e-01, -1.6008e-02, -1.8524e-03,
-2.3440e-01, -3.2373e-02, -6.7911e-02,
-1.6256e-01,  1.2316e-01,  2.7859e-02,
 8.5089e-04, -3.7401e-02, -1.8672e-02,
-1.0418e-01, -7.8407e-02, -1.8413e-02,
 8.2834e-02,  2.3128e-01,  3.2983e-02,
 3.1099e-02, -6.4485e-02, -8.1659e-02,
 1.9152e-01, -1.9609e-02,  2.7364e-02,
 1.0458e-02, -1.2507e-01,  4.1334e-02,
-4.6215e-02,  5.6944e-02,  2.1477e-02,
-1.4934e-01, -6.8383e-02,  2.7957e-02,
-3.6846e-01,  4.8766e-01,  6.4000e-02,
-3.9621e-02, -8.1667e-03,  4.5997e-02,
-6.1391e-02,  1.2976e-02, -3.2152e-02,
 7.5767e-02,  1.2931e-01, -2.3498e-02,
 4.0320e-02,  1.3876e-02,  1.1022e-02,
-6.2401e-41,  5.8564e-40,  3.9473e-40,
-5.6890e-40, -2.6022e-40, -2.9841e-40,
-4.2456e-40, -1.1546e-40,  4.4955e-40,
-4.2969e-02, -1.0995e-01,  1.3021e-01,
 1.0142e-01,  5.2225e-01, -5.5486e-02,
-7.2349e-02,  8.5470e-02,  2.3438e-02,
-1.0690e-01, -1.4370e-01, -1.2632e-01,
 2.8754e-02,  1.1662e-01,  5.6515e-02,
-1.5726e-01, -1.4945e-01, -4.4956e-02,
 1.6574e-01, -5.6894e-02, -2.0851e-01,
 8.1498e-03, -2.5441e-01, -1.4412e-01,
-1.0959e-02, -2.5811e-02,  8.8934e-02,
 6.3594e-02, -9.3314e-02,  7.8247e-02,
 4.6795e-02, -2.2774e-01,  7.1041e-02,
 1.4830e-01,  1.9911e-01,  5.1978e-02,
 7.4936e-02,  2.3104e-02,  6.3928e-02,
-1.3118e-02,  6.7544e-02,  7.9514e-02,
 2.2335e-02, -9.9442e-02,  6.8070e-03,
 2.4395e-02, -3.3576e-02,  5.5508e-02,
-4.0872e-02,  5.4501e-02, -5.7051e-02,
 8.6621e-03, -1.5361e-01,  1.2630e-01,
-2.2344e-01,  1.3335e-01, -1.1688e-01,
-2.4232e-01,  3.3319e-01, -1.2580e-01,
-2.2169e-02,  2.0594e-01,  2.6521e-02,
 4.1883e-40, -3.4540e-40,  4.9152e-40,
-1.5711e-40,  3.3927e-40, -5.5069e-40,
 5.5831e-40, -5.2011e-41,  1.0351e-40,
 1.7989e-01,  2.3787e-02,  5.7447e-03,
 4.8748e-01,  3.0152e-01,  3.5517e-02,
 2.2155e-01,  1.8812e-01,  3.0994e-02,
 7.8657e-02, -7.1135e-02, -5.8293e-02,
-1.4220e-01,  1.6004e-02, -2.5180e-02,
-1.6811e-01, -2.3441e-01,  1.4810e-02,
 5.3140e-02, -1.2904e-01, -1.5105e-02,
 5.4525e-02, -1.5418e-01,  6.6507e-02,
 8.3947e-02, -1.1975e-01,  5.3902e-02,
 8.0834e-02, -2.4321e-01, -1.0282e-03,
 3.1276e-03,  3.2495e-01, -1.3238e-02,
 4.5285e-02,  5.8777e-02, -1.3231e-01,
-6.0928e-03,  8.7145e-02,  6.2031e-02,
-5.3919e-01, -6.8810e-02, -1.0755e-01,
-2.2571e-02,  2.6237e-02, -6.8731e-03,
-6.6771e-02, -2.0586e-01,  4.7722e-02,
-3.4968e-01,  3.0912e-01,  2.4487e-01,
-4.9537e-02, -5.2779e-04,  6.7840e-02,
 1.7583e-02,  3.3222e-02, -5.7070e-02,
-2.3250e-01,  1.4470e-01, -4.9895e-02,
 3.3147e-02,  8.6319e-02,  4.4719e-02,
-6.9454e-41,  2.0308e-40, -1.1977e-40,
 5.9045e-40, -2.6129e-40,  4.8298e-40,
 4.7288e-40,  6.0736e-40,  2.2462e-40,
-4.0294e-02, -9.1437e-03, -2.4926e-02,
-2.1269e-01,  1.1602e-01,  1.4383e-02,
 5.1456e-02,  6.9047e-02,  1.6519e-02,
 6.3737e-02, -9.0181e-02,  7.0716e-02,
 7.0061e-02,  7.9046e-02, -4.3925e-02,
 7.4396e-02, -5.2797e-02,  3.8125e-02,
 7.5999e-02, -5.1307e-02,  2.4326e-03,
-3.1716e-02, -1.2567e-01, -3.3898e-02,
 8.4925e-02, -5.2404e-02,  2.8535e-02,
 9.6844e-03,  4.6980e-02,  3.8552e-02,
-5.7110e-02,  3.2163e-02,  1.5219e-02,
 6.6905e-02, -2.7934e-02,  1.4184e-03,
-2.4239e-02, -8.6317e-03, -2.3295e-03,
-2.3065e-02,  1.0076e-01,  2.1562e-03,
-1.3647e-02, -3.4262e-02,  2.5777e-02,
 7.6601e-02,  1.3654e-01,  2.1458e-03,
 1.4542e-01,  3.6310e-01,  1.6266e-01,
-5.8465e-02,  4.3751e-02,  1.9227e-02,
 9.1783e-03, -5.9547e-02, -1.8234e-02,
-5.3399e-02,  1.9218e-01, -4.6238e-02,
-1.9052e-01,  1.4635e-02,  2.9536e-02,
 1.4621e-40, -5.5132e-40, -4.6215e-40,
 4.3948e-40, -2.7285e-40, -5.5709e-40,
 1.9428e-41, -4.0333e-40, -5.4469e-40,
 9.3126e-02, -1.3236e-01,  9.9350e-02,
-1.3308e-01,  3.5030e-01,  9.2221e-02,
 1.1783e-01,  1.6648e-01, -7.9150e-02,
 2.2654e-01, -1.2546e-01, -1.2354e-01,
-1.6457e-01, -6.0740e-02, -3.1069e-02,
-8.3203e-02, -1.8064e-01,  4.6900e-02,
 1.2059e-01, -1.0569e-01, -7.1196e-02,
-9.2991e-02, -1.7587e-01,  1.3100e-03,
-1.5492e-01, -1.3849e-01,  1.2245e-01,
-5.5276e-02, -9.7867e-02,  3.5550e-02,
-6.0264e-02,  4.7760e-02,  6.0242e-02,
-5.4096e-03,  2.4646e-01,  6.3592e-01,
 5.8559e-02,  6.1117e-02,  8.0334e-02,
-4.4582e-03, -1.2028e-01,  8.7394e-02,
-2.5880e-02, -1.2206e-01,  1.2199e-01,
 4.1990e-02, -1.3283e-01,  4.9047e-02,
-4.9532e-02,  2.7688e-01, -4.6064e-03,
-2.8812e-03, -2.4404e-01,  5.8614e-02,
-1.4262e-01, -1.2810e-03, -1.2060e-01,
-8.3595e-02,  5.6532e-02, -7.7556e-02,
-1.3364e-01, -1.3883e-01, -1.2335e-01,
-1.3273e-40,  6.5184e-41, -4.6946e-40,
-4.0031e-40, -1.2807e-40, -3.1584e-40,
 1.3009e-40,  2.4187e-40, -1.4202e-40,
-8.8844e-03,  1.0101e-03, -6.0190e-02,
-1.8851e-01, -7.6662e-02, -1.4562e-01,
 2.9983e-02, -8.1533e-02,  1.1256e-02,
 1.0205e-01,  6.7850e-02, -1.0911e-01,
-1.2846e-01, -5.4605e-02,  6.2182e-02,
-1.0797e-01, -5.1281e-02, -1.2036e-02,
-8.1693e-02, -7.0432e-02,  1.6990e-01,
-1.7329e-01, -2.2084e-01, -3.0977e-02,
 8.2771e-02, -3.3089e-01, -1.4842e-01,
 1.9576e-02, -1.5953e-01, -1.0348e-01,
 6.6014e-02,  6.0094e-01, -6.9891e-04,
 7.4969e-02, -1.4250e-01,  4.3221e-02,
 1.6796e-02, -6.8125e-03,  4.7028e-02,
-3.3421e-01, -2.2987e-01,  4.2936e-02,
 9.3985e-04,  9.0827e-02,  2.4211e-01,
-8.1571e-02, -1.0276e-01,  1.9092e-01,
 2.1112e-01,  2.6837e-02, -2.5822e-01,
-1.3290e-01,  1.6135e-01, -2.7672e-02,
 3.4465e-01, -8.3286e-03, -6.1936e-02,
 2.7406e-01, -6.8357e-02,  1.7426e-01,
-9.0872e-02,  1.2999e-01,  7.2366e-02,
 3.0944e-40, -1.2808e-40,  2.9336e-40,
 5.5561e-42,  3.0978e-40,  1.0027e-40,
-1.5881e-40, -2.9858e-40,  3.1599e-41,
-9.1935e-02, -2.2666e-04, -6.2821e-02,
-1.8605e-01,  3.0238e-01,  3.2759e-02,
-5.0771e-02,  1.4585e-02, -1.0872e-01,
 2.5511e-02, -9.3394e-02,  1.4810e-02,
-6.2906e-02,  9.2472e-02,  1.2845e-02,
-2.9041e-01, -9.6489e-03, -2.7277e-02,
-6.9896e-02, -1.1645e-01, -5.9870e-02,
-2.8037e-02, -2.2649e-01,  5.1781e-02,
-1.4588e-02,  4.8753e-02, -2.8256e-02,
-1.6462e-02,  8.0795e-02,  3.6222e-02,
 8.0392e-02,  3.0118e-01,  2.0021e-01,
 1.0394e-01,  6.4196e-01,  4.9545e-01,
 2.1242e-02, -1.2514e-01,  1.0066e-01,
-4.7676e-02, -2.0736e-02, -5.6951e-03,
-8.3021e-02,  4.6763e-02,  1.7551e-01,
 2.0038e-02,  1.8084e-01,  1.3244e-02,
 1.0280e-02,  2.8740e-01,  8.9837e-03,
-2.9437e-02, -3.7366e-01, -1.1861e-01,
-4.8248e-03, -1.2970e-01, -1.8680e-02,
 1.8458e-01,  5.6509e-02,  1.2734e-01,
 1.9423e-01, -3.6960e-01, -2.5555e-02,
 6.7959e-41, -3.2251e-40, -3.0631e-40,
-4.0701e-40,  9.7399e-41,  2.2917e-40,
 2.0169e-40,  5.7891e-40, -4.1286e-40
}
,)"
R"(
{
 5.6253e-02,  1.0118e-02, -8.2749e-02,
-6.4074e-02,  4.0723e-02,  1.1657e-02,
-1.1560e-01, -3.5596e-03, -2.6713e-02,
-7.9090e-02, -2.9223e-01,  1.5759e-01,
 6.8756e-02,  1.5738e-01,  1.5413e-01,
-6.1288e-02, -1.2536e-01, -1.5966e-01,
 1.1165e-01,  5.0211e-02, -1.0338e-01,
-5.2364e-04,  1.7660e-01, -2.2504e-03,
-1.7697e-01,  1.8500e-02,  2.0693e-02,
-2.5907e-02, -1.4201e-01,  8.4467e-02,
 1.1138e-02,  2.1769e-01, -4.2422e-01,
 6.5046e-02,  2.6834e-02,  2.9047e-03,
-1.2130e-01, -5.1773e-01, -8.0393e-02,
 3.0204e-02,  3.5952e-01,  1.6681e-01,
-9.4720e-04,  7.7291e-02,  8.3039e-02,
 3.4689e-01, -1.2389e-01, -2.0666e-01,
-2.9650e-02,  1.1102e-01, -1.4782e-01,
 3.2193e-02, -3.9862e-02,  1.6440e-02,
-8.4264e-02,  1.0192e-01, -6.4256e-02,
 2.2950e-02, -6.6511e-02, -6.3814e-02,
 4.3744e-02, -1.0557e-01, -1.2045e-02,
 1.6330e-01,  6.6130e-01,  1.5497e-01,
 1.7103e-01,  1.5073e-01,  1.7400e-01,
 9.0985e-04,  1.0917e-02, -1.3322e-02,
-6.4273e-02, -6.2178e-02, -7.7223e-02,
-1.0332e-01, -2.1072e-01, -2.2843e-03,
 3.2717e-02, -6.3754e-02,  5.0359e-02,
-5.2566e-02,  6.2090e-02, -1.5614e-02,
 1.4570e-02, -1.0243e-01,  1.3091e-01,
-2.9988e-02, -7.5897e-02, -9.4541e-04,
-2.7999e-01, -4.7415e-03,  5.6419e-02,
 7.0565e-02, -4.9273e-01, -1.2936e-01,
 5.5685e-02, -5.8924e-03, -3.1967e-02,
 8.8602e-02,  2.9337e-01,  1.3753e-01,
 1.0063e-02,  1.6348e-02,  1.0063e-01,
 3.6230e-02,  1.7968e-02, -1.1624e-01,
-2.2488e-02,  1.3474e-01, -1.1419e-01,
 2.8576e-02, -7.4794e-02, -7.7261e-02,
 5.8874e-02, -2.9448e-03,  6.0207e-02,
 1.4642e-01,  1.2321e-01, -2.4936e-01,
 2.2609e-02, -2.8171e-01,  1.1510e-01,
 2.6056e-02, -2.7532e-02, -4.7505e-02,
-2.8762e-02, -1.2610e-02, -8.3766e-02,
-5.0992e-02, -5.7269e-03, -7.0981e-02,
-9.6191e-02, -9.2384e-02, -5.3328e-02,
 2.3989e-01,  3.9819e-01,  1.8451e-01,
 3.6888e-02,  1.1023e-01,  4.4804e-03,
-4.4140e-03, -4.8275e-03,  2.0018e-02,
-2.4346e-02, -6.5546e-02, -4.6065e-03,
 2.2298e-01,  2.8810e-01,  1.4071e-02,
-1.7315e-01, -5.7961e-02, -9.9136e-02,
 3.6456e-02, -1.5518e-02,  6.4490e-02,
 4.6983e-02,  5.2743e-02,  3.0802e-01,
 6.7940e-02,  5.8777e-03,  3.1155e-01,
 9.9510e-02,  2.7974e-02, -6.6716e-02,
 3.7042e-01,  2.0813e-01, -3.1581e-02,
 7.9064e-02, -1.3699e-01, -4.4722e-02,
-8.4753e-03,  8.0676e-02,  1.5771e-01,
-1.1467e-01,  5.6269e-02,  1.1369e-01,
-1.4727e-02,  3.7263e-02, -2.0554e-01,
 8.3383e-02,  4.5848e-02, -1.1732e-02,
 4.5494e-02, -2.1406e-01,  6.0591e-02,
 4.6503e-02, -1.0362e-01,  3.8794e-02,
-4.6633e-01,  1.4504e-01,  1.4999e-01,
 2.9642e-01, -4.8807e-01, -1.6012e-01,
 1.6708e-01,  9.5313e-02, -7.5981e-02,
-4.2655e-02,  9.2470e-02, -7.7242e-02,
-2.1021e-01,  1.2423e-01,  1.4967e-02,
-5.4129e-02,  7.4355e-02, -4.7068e-02,
-1.6048e-01,  9.8742e-02,  4.4282e-02,
-6.0187e-02,  1.9495e-01,  8.3291e-02,
-7.5190e-02, -6.8429e-02,  3.7391e-02,
 5.1413e-04,  1.5098e-01, -1.1549e-01,
 1.6875e-01,  1.8040e-01, -1.3162e-01,
 7.7101e-02,  2.0816e-01,  7.6289e-02,
-1.7528e-02,  1.4408e-02,  3.7500e-02,
 3.8647e-02,  1.6850e-01,  1.7535e-02,
-2.8205e-02,  1.0273e-02,  1.6688e-01,
 4.3676e-02,  6.9895e-02,  8.1063e-03,
-2.6117e-01, -1.0920e-01,  5.2209e-02,
-5.2749e-02, -1.7062e-02, -9.6808e-02,
 2.7324e-02,  9.1342e-02, -5.0968e-02,
 1.0689e-01,  5.0565e-01,  4.6004e-01,
-6.6862e-03,  3.4162e-03,  3.3559e-01,
 3.5084e-02,  1.9123e-02,  1.0073e-02,
 1.6995e-01,  3.4099e-01, -4.0847e-01,
-5.5317e-03,  4.0230e-02, -2.0305e-01,
-8.9786e-02,  1.9667e-01,  3.8111e-02,
 3.0607e-02, -1.9084e-02, -6.5114e-02,
 8.5394e-02, -1.3992e-01,  1.4988e-02,
-1.5926e-02, -9.1200e-03, -7.2328e-02,
 1.3548e-01,  7.1040e-01, -9.4208e-02,
 2.5411e-03, -7.2159e-02,  1.0848e-01,
-8.9029e-02, -8.6339e-02, -2.7546e-02,
 6.0378e-02,  2.8401e-01, -6.6550e-02,
-3.0486e-02,  5.0307e-02, -1.1084e-02,
 2.9732e-02,  9.9960e-02, -7.7408e-02,
 3.4940e-01, -5.6048e-01,  2.9053e-02,
-2.6991e-02,  4.9637e-02, -3.9322e-02,
-1.0418e-02,  1.0931e-01, -6.1609e-02,
 3.6057e-02,  9.3866e-02, -1.0339e-01,
-1.8572e-02, -2.0889e-02, -7.4531e-02,
-7.3236e-02, -4.5908e-02,  2.2705e-02,
-1.5148e-02,  2.1735e-01,  2.2477e-02,
-3.4153e-02, -2.6939e-02, -5.0167e-03,
 6.6774e-02,  2.0168e-01, -7.5083e-02,
 5.6608e-02,  2.2799e-01, -3.7473e-01,
-7.2336e-02,  4.4329e-02, -3.6747e-02,
 3.5355e-02,  1.8671e-01, -4.0167e-02,
 1.2871e-01,  3.5050e-01,  1.8090e-01,
-6.2429e-02,  6.2184e-02,  6.8804e-02,
-8.0164e-02, -2.4387e-02, -5.0309e-03,
 1.0089e-01, -3.0008e-02,  1.7251e-02,
-9.4662e-03, -1.4760e-02,  7.3434e-03,
 7.3290e-02,  2.2546e-02, -2.9015e-02,
 7.9944e-02, -2.6972e-01,  7.1349e-02,
-1.7026e-02,  1.1461e-01, -4.1288e-02,
-5.3732e-02, -2.4618e-01, -1.2890e-02,
 8.6133e-02,  1.9503e-01,  8.2202e-02,
-1.0060e-03, -4.5931e-04, -1.8789e-02,
-4.0843e-02, -7.8149e-03, -6.1464e-02,
-7.9364e-02, -5.9647e-02, -5.4059e-03,
 1.9553e-01, -2.4079e-01, -7.9538e-03,
 5.3620e-02,  1.4198e-01,  6.5651e-03,
 2.3512e-02, -2.6609e-02, -4.6435e-02,
 1.2499e-02,  5.1079e-02, -2.2713e-02,
-7.1554e-02,  1.0608e-01,  5.8972e-02,
 1.8638e-01, -2.1053e-01, -6.4009e-02,
 1.0851e-01,  7.2187e-02,  8.9722e-02,
-4.5365e-04,  1.0826e-01, -6.4141e-02,
-2.3874e-02, -4.6307e-02, -2.7813e-02,
 1.8385e-02,  9.4687e-02,  6.8374e-02,
 9.4526e-02,  1.4432e-02,  1.5937e-01,
 1.1292e-01, -3.4274e-01, -1.0813e-01,
-7.4636e-03,  3.7101e-02,  3.7226e-02,
 3.7079e-02, -3.9169e-02, -3.7752e-02,
-7.9021e-02,  8.5978e-02,  1.0958e-02,
-5.8576e-02,  5.5931e-02,  4.8301e-02,
-1.3402e-01, -3.3809e-01, -4.4369e-02,
 1.4262e-01,  6.5254e-02, -3.3366e-01,
 1.2416e-02, -9.0492e-02, -5.8205e-02,
-1.4886e-01,  4.0598e-02, -1.4219e-01,
 2.0223e-03, -2.8673e-01, -3.3622e-01,
 1.9191e-02, -2.2104e-02,  1.9048e-02,
 6.0021e-02,  2.2520e-01, -5.3972e-02,
 1.6226e-01, -2.1918e-01, -5.2117e-02,
-6.2363e-03,  2.0266e-01, -7.3323e-03,
 1.1137e-01, -1.9300e-02, -5.4983e-02,
-1.8338e-01,  6.2511e-01, -1.7909e-01,
 1.7003e-01,  1.7902e-01,  5.4462e-02,
 5.6847e-02, -7.4696e-02, -1.1354e-02,
 1.0544e-01, -1.4918e-01,  4.8208e-02,
-5.6262e-02, -2.3303e-01, -2.9916e-02,
-3.3261e-02,  1.3287e-01,  1.9831e-02,
-1.3907e-01, -1.6180e-01, -7.2323e-03,
-5.1689e-02,  6.3121e-02, -1.4480e-01,
 1.1143e-01,  4.9625e-02, -5.4369e-02,
-3.9247e-01,  2.3412e-01, -3.6726e-02,
-1.1468e-02,  3.4045e-02,  6.6454e-02,
-5.0103e-02,  6.1740e-02,  4.2922e-03,
 1.7669e-01, -8.1250e-03,  6.3694e-03,
-6.7723e-02,  7.4576e-02,  1.0113e-02,
 1.1264e-01, -4.4691e-02, -5.3575e-02,
 3.4691e-02, -1.2201e-02, -8.4221e-02,
 2.3677e-01,  3.9073e-01,  2.4710e-02,
-8.4580e-02, -1.0747e-01, -6.5695e-02,
 1.5386e-01,  1.4041e-01,  6.9961e-03,
 2.6138e-02,  2.3149e-02, -1.8820e-02,
-3.3541e-02,  3.2089e-02, -1.8916e-02,
 1.0564e-01, -7.5319e-02, -5.4282e-02,
-6.9388e-03, -2.0873e-02,  5.6100e-02,
 2.3524e-02, -6.4296e-02,  5.8950e-02,
-3.1415e-03, -4.1203e-02,  1.0781e-01,
 1.7848e-02, -2.9535e-02, -1.6412e-02,
-4.6649e-02,  8.1277e-02, -5.9918e-02,
 8.1522e-02, -9.2037e-02,  8.1039e-03,
-6.5541e-02,  5.1811e-02, -1.4380e-03,
 5.0419e-02,  9.3091e-03, -2.8054e-02,
-3.0979e-02, -2.5366e-02,  3.5265e-02,
-3.7730e-02,  5.7574e-02,  3.4683e-02,
 4.8819e-03, -2.9519e-02,  3.7740e-02,
 6.4546e-02, -3.7272e-01, -8.5393e-02,
-3.0223e-02, -7.7899e-02,  2.7365e-03,
 2.2282e-02, -3.3440e-02,  1.9048e-02,
 2.3275e-02, -2.1153e-02, -2.0385e-02,
-4.6245e-02,  2.2443e-02, -3.0206e-02,
-2.5302e-02, -1.1418e-02,  4.8228e-02,
 5.8367e-02, -4.3062e-02,  2.2814e-02,
-4.6279e-02,  5.0052e-02,  2.2961e-02,
-5.4984e-02,  1.4773e-01, -2.5546e-02,
 3.3025e-02, -1.0138e-01,  6.3886e-02,
 1.2403e-02,  1.6215e-02,  1.0783e-02
}
,)"
R"(
{
 2.5042e-02, -5.3266e-02,  3.8484e-02,
 3.7189e-03,  1.0493e-01,  1.4459e-01,
-3.7442e-02, -1.5744e-01,  1.9957e-01,
-1.9203e-02,  1.6256e-02,  4.2906e-03,
-3.1637e-02,  5.0287e-01, -6.9504e-02,
 1.4677e-03, -8.9984e-02, -9.0376e-02,
 4.0578e-02,  2.4004e-02,  3.4044e-03,
 7.5916e-02, -1.3564e-01, -9.0296e-02,
 3.4156e-02,  7.2494e-02, -2.0037e-02,
-6.4614e-02, -1.7301e-03, -3.3444e-02,
-2.7950e-01,  7.1351e-01,  4.2825e-02,
 2.4797e-02,  5.4162e-04, -8.9676e-02,
 3.8002e-02, -2.7692e-02, -1.7757e-02,
 1.9356e-01,  1.9598e-02, -1.0862e-01,
 2.5734e-02,  1.1703e-02, -7.3912e-02,
-6.0213e-04,  1.6024e-01, -6.4591e-03,
 3.1779e-02, -3.1049e-01,  1.2684e-02,
-1.0098e-01, -1.8839e-01,  5.1387e-02,
 5.2004e-02,  3.1489e-01,  5.9716e-01,
-7.2238e-02,  3.4332e-01, -2.0655e-01,
 1.1013e-03, -5.0328e-02, -4.6118e-02,
 9.4442e-04,  2.7964e-02,  1.7672e-02,
-8.6022e-02, -3.8280e-02,  2.8017e-04,
 3.3824e-02, -6.7883e-02,  1.0529e-02,
-6.5982e-02,  1.1385e-01,  3.0091e-03,
 1.2330e-01,  6.1876e-01,  5.7145e-02,
-4.3835e-02, -6.8186e-01, -1.0917e-01,
 3.2006e-02, -2.0627e-03, -6.9043e-02,
 7.2219e-02, -3.2393e-01, -2.6657e-02,
 1.3523e-02,  1.8099e-01,  4.9168e-02,
 7.1367e-02,  9.8283e-02,  1.0425e-01,
 2.2286e-01, -5.9374e-01,  1.0014e-01,
 6.5700e-02,  1.3618e-02, -7.4045e-02,
 1.0481e-01,  3.0734e-02,  1.0431e-02,
-2.1314e-01, -7.2817e-02,  1.2036e-01,
-5.4180e-02,  1.0500e-01,  2.7821e-02,
-5.0657e-02,  8.7702e-02,  7.0234e-02,
 9.0349e-02,  1.4905e-01,  1.1612e-01,
 5.9924e-02,  2.4928e-01,  1.7078e-01,
-5.9110e-02, -7.4252e-02,  9.8241e-03,
-1.2006e-01,  1.3879e-01, -1.4322e-02,
-7.5463e-02,  1.4407e-02, -6.9202e-03,
 7.0279e-02,  1.7065e-01, -2.5150e-01,
-2.6289e-02,  3.8421e-01, -2.2051e-01,
-2.8918e-02,  4.0074e-02, -7.1296e-02,
 1.0357e-01, -1.8885e-01,  2.3780e-02,
-1.8884e-01, -4.3326e-01, -1.1465e-01,
 3.3497e-02, -1.3462e-01, -3.4127e-02,
-1.2731e-02,  5.4326e-02, -2.6581e-02,
 5.1753e-02,  6.8200e-03,  4.3246e-03,
-6.9963e-02, -1.5618e-01,  2.5192e-01,
 2.2890e-02,  6.1421e-02,  5.2832e-02,
-9.8369e-02, -1.1452e-01,  1.7420e-01,
 2.0392e-01, -1.1322e-01,  9.8462e-02,
-3.3547e-02, -2.8993e-01,  7.0080e-02,
 8.2478e-02, -1.9881e-01,  1.2849e-01,
-2.7802e-01, -1.5621e-01,  6.2712e-02,
 1.3028e-02,  1.4716e-01,  2.0434e-02,
-4.4071e-01,  3.8359e-01, -1.6655e-03,
-2.0297e-01,  1.5631e-01,  7.7086e-02,
 9.6714e-03, -5.5842e-03,  7.9155e-03,
 1.4525e-01, -3.2228e-01,  1.1454e-01,
 1.4527e-01, -3.0399e-02, -6.7043e-02,
 9.4233e-03, -1.1296e-02, -1.0927e-01,
 7.9300e-02,  5.5286e-02, -1.1558e-01,
 3.8173e-01, -5.4351e-02, -1.7890e-01,
 5.4882e-02,  1.5119e-01,  1.8363e-01,
-8.8223e-02, -9.0083e-02,  4.8221e-01,
 4.0890e-02,  5.6429e-02, -2.8538e-01,
 1.2102e-02, -1.8177e-02, -3.1643e-03,
-6.9064e-02,  3.1853e-04, -7.0113e-02,
 9.7308e-02,  1.0691e-01, -6.5919e-02,
-1.4536e-40, -1.7049e-40, -2.6781e-40,
 4.5792e-40,  1.4489e-40,  1.3645e-40,
-5.8774e-40, -2.2505e-40, -4.7571e-40,
 3.3670e-40,  1.5398e-40, -3.3819e-40,
 2.6303e-40, -1.9434e-40, -5.5555e-40,
-4.3830e-40, -2.8750e-40, -3.0788e-41,
 5.6364e-40,  3.1307e-40, -2.3064e-41,
 2.8909e-40, -5.8115e-40,  2.9852e-41,
-1.9273e-40, -7.5503e-41, -6.0335e-40,
 5.8073e-40,  2.9252e-40, -1.3038e-40,
 5.2260e-40,  3.8172e-40, -2.0389e-40,
-2.1905e-41,  1.8473e-40, -2.9226e-40,
 2.9957e-41,  2.6068e-40,  6.1324e-40,
-4.3013e-41,  5.1421e-40, -4.1157e-40,
 2.1416e-41, -1.6614e-40, -3.0843e-42,
-4.3402e-40,  2.8507e-40,  1.1560e-40,
 3.8826e-40, -3.0797e-40, -6.0685e-40,
 5.4170e-40, -6.1858e-40,  9.3049e-41,
-1.9491e-40, -1.9211e-40, -6.2723e-40,
 3.9906e-40,  1.2356e-40,  3.8682e-40,
 2.8630e-40,  6.2303e-40,  5.3034e-40,
-4.1904e-40,  4.8916e-40, -3.6125e-40,
-5.5393e-40, -2.4980e-40, -6.1877e-40,
 2.7289e-40, -1.8348e-40, -5.6663e-40,
 2.5152e-02, -3.2878e-02,  2.1626e-02,
 1.9879e-01,  2.9080e-02, -3.0331e-03,
-2.3380e-01, -2.3578e-02,  1.1871e-01,
-3.1824e-02, -5.5095e-02,  3.1338e-02,
-3.2199e-02, -4.3820e-01,  4.1391e-02,
-4.1207e-02,  3.7475e-01, -1.8548e-01,
-1.4460e-02, -8.7834e-02, -3.2343e-02,
 2.4023e-01,  7.1916e-01, -1.8559e-01,
-6.7635e-03, -9.4409e-02, -1.7890e-02,
-5.8334e-02,  1.8886e-01,  6.1547e-02,
-2.6152e-01,  6.6722e-01, -1.2486e-01,
-4.8128e-02,  1.0510e-01, -4.2619e-02,
 3.0101e-03,  9.6380e-02,  6.6140e-02,
 1.0201e-01, -2.3240e-01, -1.8356e-01,
 4.0019e-02,  2.2985e-01, -1.2980e-01,
-1.1400e-01, -1.9221e-01, -3.4158e-02,
 2.2871e-02, -6.8684e-01, -1.0856e-02,
 2.6311e-02,  2.5422e-02, -1.5190e-02,
 3.2182e-02, -5.6346e-02,  3.2655e-02,
-1.6912e-02,  8.4264e-02, -7.9521e-02,
 1.2788e-03, -7.1110e-02,  8.6585e-02,
-4.2829e-02,  1.0778e-01, -6.8129e-02,
 5.8156e-03, -2.3998e-01,  1.9052e-01,
-4.1855e-02,  1.0140e-01, -1.7139e-02,
 5.2301e-40, -2.9923e-40,  3.8688e-41,
 3.1575e-40,  1.1504e-40,  5.5655e-40,
-3.4499e-40,  2.3050e-40, -6.3766e-41,
 1.3282e-40,  4.5849e-40,  3.5308e-40,
-2.6657e-41,  5.9829e-40,  3.2791e-40,
-2.8348e-40,  2.5810e-40,  5.5791e-40,
 4.2613e-40,  3.2607e-40, -2.0789e-40,
-3.9054e-40, -2.5608e-40, -2.7638e-40,
 4.5027e-40,  2.7065e-40, -4.5593e-40,
 1.6336e-40, -2.0391e-40, -5.9017e-41,
-7.9899e-41, -2.9870e-40,  5.6390e-40,
-2.5560e-41, -1.9786e-40,  9.4700e-41,
-7.4049e-41, -2.3902e-40, -2.8497e-40,
-1.8912e-40, -1.5589e-40,  5.5463e-40,
-2.1782e-40, -1.9532e-40, -2.3785e-40,
 2.7539e-40,  4.0214e-40,  2.0732e-40,
 7.0120e-41, -4.4200e-40,  7.3787e-41,
 2.6452e-40,  1.1970e-40,  2.8298e-40,
 5.2721e-40,  1.9304e-40, -3.8489e-40,
-3.9759e-40,  2.6184e-40,  1.2594e-40,
 1.5831e-40,  3.7179e-40, -3.4915e-40,
-1.7681e-40, -6.9657e-41, -4.0746e-40,
 8.0894e-41,  1.6950e-40, -1.0574e-40,
-1.0590e-40,  2.8466e-41, -2.7558e-40,
-5.4027e-40,  4.4355e-41, -3.2144e-40,
-4.8838e-41, -3.8595e-40,  2.5064e-40,
 4.0365e-40, -1.0195e-40,  4.8356e-40,
 4.4499e-40, -4.4871e-40, -2.4561e-40,
 4.1687e-40,  5.2239e-40, -5.7603e-41,
-1.5211e-40, -3.5768e-40,  3.6385e-40,
 1.6089e-40,  4.1624e-40,  4.5114e-40,
 1.6438e-40, -3.6331e-40,  6.4961e-41,
 5.0899e-40,  6.1036e-40,  2.4828e-40,
 5.8681e-40, -5.7259e-40, -1.5371e-40,
 5.2654e-40,  4.7412e-40, -2.0265e-40,
-4.8621e-41,  4.9497e-40,  3.0176e-40,
 4.2235e-40,  4.5381e-40,  4.6501e-40,
-1.6124e-40, -1.9449e-40,  5.1497e-40,
-1.2891e-40, -1.6549e-40,  4.8348e-40,
-2.0735e-40,  1.3423e-41, -4.4109e-40,
-5.4218e-40, -1.1537e-40, -1.1664e-40,
 5.6006e-40,  3.4109e-40, -3.1434e-40,
 3.4969e-40, -5.3459e-40,  3.9245e-41,
 2.4028e-40,  5.7774e-40, -6.2973e-40,
 1.8802e-40, -4.6258e-41, -5.0716e-40,
 3.4962e-40, -6.2313e-41, -2.7290e-40,
-5.2709e-40, -3.2225e-40,  2.4245e-40,
-3.6300e-40, -2.0794e-40,  4.0541e-40,
-3.5157e-02,  6.8337e-02,  1.6149e-02,
-5.8650e-03,  6.0605e-01,  3.1738e-02,
 9.3306e-02,  2.1499e-01,  1.3609e-01,
 6.4043e-02, -1.0253e-02, -6.2813e-04,
 4.6828e-02, -3.9619e-01, -9.2633e-03,
-8.1752e-02,  9.9083e-02,  4.4296e-03,
 7.1594e-02,  3.9860e-02,  8.1088e-02,
 1.7750e-01, -1.2381e-01,  1.4476e-01,
 2.3416e-02,  1.2819e-01,  1.0816e-02,
 5.5296e-02,  5.5199e-02, -2.1253e-02,
 1.7214e-01,  2.0542e-01, -3.7859e-03,
 1.2831e-01,  3.2087e-02, -5.1851e-02,
-2.3686e-02,  1.2271e-01, -1.6009e-02,
-2.0176e-01,  7.4757e-01, -3.4526e-02,
-4.7055e-02, -3.7099e-01, -1.9216e-01,
-8.8030e-02, -2.5853e-02, -1.7087e-02,
-2.0533e-01,  1.5214e-01, -1.8639e-03,
-1.1236e-01, -2.4612e-01,  6.3094e-02,
 2.3829e-02, -5.0078e-03,  5.3854e-02,
-9.6934e-03,  3.7047e-02,  4.7325e-01,
 5.6975e-03, -8.6108e-02,  6.5569e-02,
-3.9768e-03,  2.0580e-02, -4.1931e-02,
 6.9577e-02, -1.0416e-01, -2.5037e-03,
-1.9198e-02,  6.2027e-02, -1.0833e-02
}
,)"
R"(
{
-5.3430e-40,  2.5717e-41,  5.7504e-40,
 7.1679e-41,  6.2076e-40, -8.4201e-41,
-4.2111e-40,  3.4851e-40,  1.3009e-40,
 3.3016e-40, -7.6473e-41, -1.8392e-40,
 2.2773e-41,  1.2087e-40,  1.1565e-40,
 6.5190e-41,  2.0075e-40,  2.5796e-40,
 5.0575e-40, -2.6261e-40, -2.5486e-40,
-3.9886e-40, -6.0644e-40,  2.9264e-40,
 8.9627e-41, -3.0550e-40, -2.3456e-40,
-4.8855e-40, -4.8867e-40, -5.0492e-40,
-1.0706e-40,  5.3827e-40, -1.6413e-40,
 1.4714e-40, -3.4024e-40, -4.4881e-40,
 3.2361e-40,  2.0858e-40,  3.8836e-40,
 2.0949e-40,  5.9633e-40, -1.7878e-41,
-4.1980e-40, -4.4383e-40,  2.7859e-40,
 7.0317e-42, -8.9973e-41,  5.8700e-41,
 1.8411e-40, -3.6097e-42,  2.7362e-40,
 5.4341e-40,  6.0305e-40,  5.9004e-40,
 5.2692e-40, -6.3449e-41,  1.2075e-40,
 7.5297e-41,  8.9267e-41,  4.9139e-40,
-1.4609e-40,  3.1821e-41,  2.3288e-40,
 3.1748e-41, -3.8052e-40, -2.4322e-40,
-5.7959e-40,  6.1966e-40,  3.4964e-40,
-5.6776e-40, -6.8327e-41, -3.3777e-41,
-5.9108e-02,  3.5468e-02, -2.8772e-02,
 6.8602e-01,  1.4232e-01,  1.1954e-02,
-3.8234e-02,  7.1837e-02, -1.8832e-02,
 4.7972e-02,  1.1623e-02, -2.1687e-03,
-4.9744e-01,  2.7751e-01,  1.7862e-02,
 7.4286e-02,  3.1309e-03,  1.1030e-03,
-6.1084e-01, -8.5679e-03,  9.4956e-03,
-4.5246e-01, -1.2126e-01, -3.7368e-02,
 2.5624e-02,  1.2087e-02, -1.5431e-02,
 6.0313e-40,  1.8404e-40, -7.2006e-41,
 6.0697e-40, -9.1199e-41,  5.8965e-40,
 5.4830e-40,  1.3014e-40,  1.5585e-41,
-3.6027e-02, -6.3004e-03,  1.5237e-02,
 6.0743e-01,  9.2523e-02, -4.7370e-03,
 3.4407e-02, -8.3823e-02,  1.6898e-02,
 5.7527e-40, -5.0621e-40, -2.9035e-42,
 3.8199e-40, -2.2913e-40, -5.0895e-40,
 4.0079e-40,  5.1744e-40, -3.3006e-40,
 6.1448e-40,  1.2347e-40, -3.1673e-40,
 7.3214e-41,  5.2143e-40, -2.6071e-40,
 1.6109e-40, -2.0298e-40,  9.5817e-41,
 6.9876e-02, -2.9290e-02,  3.2294e-03,
-4.2632e-01,  1.5789e-01,  3.6809e-02,
 2.1220e-02,  1.6531e-04,  6.8502e-03,
-6.5221e-02,  8.8059e-02,  5.7934e-03,
-1.7280e-01,  1.5303e-01,  1.7663e-01,
-1.2908e-01, -1.1749e-01,  5.7887e-02,
 1.0685e-01,  2.2763e-01,  3.3796e-02,
 1.7629e-01,  3.8882e-01,  6.3540e-02,
 6.4707e-02,  1.0046e-01, -8.1911e-02,
-3.9718e-03,  4.6416e-02,  4.7357e-02,
 7.3694e-02, -1.6444e-01,  2.4784e-02,
-3.0808e-03,  2.7399e-02, -2.9216e-04,
 2.4428e-40, -3.0160e-40,  2.3184e-40,
-4.9114e-40,  5.6685e-40, -3.6020e-40,
 2.2618e-40, -2.8145e-40,  2.1149e-40,
 2.3559e-02, -8.6949e-02, -3.8350e-02,
-2.9547e-01,  7.0187e-01, -8.3979e-02,
-2.8576e-02, -1.6538e-01, -5.2465e-02,
-1.6016e-40, -1.4760e-40, -2.1977e-40,
 4.3180e-40,  4.1724e-40, -1.2969e-40,
-1.3023e-40, -1.0095e-40, -1.5965e-40,
-4.0721e-40, -4.1747e-40, -4.3706e-40,
-4.2838e-40, -4.5507e-40, -4.6023e-40,
-3.7435e-40, -3.9889e-40, -4.2249e-40,
-1.2429e-01, -3.5062e-01, -1.1418e-01,
-4.0787e-02,  6.1690e-01, -1.0085e-01,
 1.6098e-02,  8.5100e-02, -1.1621e-02,
 3.0709e-40, -4.4880e-40, -2.7530e-41,
-1.2649e-40, -5.3936e-40,  5.0995e-41,
 4.4003e-40, -2.1211e-40, -6.6422e-43,
-1.8989e-40, -3.6631e-40,  4.1392e-40,
-3.9057e-40, -5.5599e-40,  6.9979e-41,
 3.8983e-40,  5.6737e-41,  2.3997e-40,
-9.4862e-41,  2.4256e-40, -3.7040e-40,
 1.6374e-40,  3.5439e-42, -1.0385e-40,
 3.6145e-40, -2.4342e-41, -3.0115e-40,
-6.0009e-40, -5.2386e-41, -1.2504e-40,
 2.9237e-40, -1.2290e-40, -1.1502e-40,
-3.5887e-40, -6.1810e-40, -1.6289e-41,
 2.5438e-41,  5.1229e-40, -2.4915e-40,
 1.3516e-40,  3.3553e-40,  8.5831e-41,
-8.5122e-41,  3.7625e-41,  2.5507e-40,
-1.5828e-40,  2.1991e-40, -1.5628e-40,
-5.3110e-40,  5.1395e-40, -5.8162e-40,
-3.1571e-40, -5.5139e-40,  1.2299e-40,
 4.8855e-40, -9.3940e-41, -6.2534e-40,
-3.3275e-40, -2.4982e-40, -1.2956e-40,
-6.0047e-40, -1.8712e-41, -7.3274e-42,
-2.8519e-40,  3.5541e-40,  2.4485e-40,
-8.1435e-41, -2.7091e-40,  7.1206e-41,
-5.9519e-41, -2.5552e-40, -3.6189e-40,
 7.7038e-02, -1.6317e-02, -2.4118e-02,
-4.3086e-02, -2.1512e-01,  1.2288e-01,
 1.8237e-01, -1.5438e-01, -1.1346e-01,
-4.6141e-02, -4.0750e-02, -5.6414e-04,
-1.5640e-01, -3.4506e-01, -1.4441e-02,
-2.0278e-01, -3.1403e-01, -6.2542e-02,
-1.9622e-02,  1.6348e-02,  6.9859e-03,
-9.3142e-02,  1.0368e-02, -5.6585e-02,
 8.4213e-02,  1.0776e-01, -1.0315e-01,
 8.7873e-41, -5.3947e-40,  1.1714e-40,
 7.5534e-41, -1.1871e-40, -5.4012e-40,
 3.8269e-41, -1.4913e-40, -3.1802e-40,
-3.4707e-02,  1.2518e-02,  9.4679e-03,
 1.2254e-01,  1.9394e-01,  2.6530e-02,
 2.2413e-01, -1.6298e-01, -6.1446e-02,
-1.1042e-42, -2.7255e-40, -5.5067e-40,
 3.8272e-40,  4.9956e-40, -3.2074e-41,
 2.8351e-40,  4.2501e-40,  3.9389e-41,
 6.1941e-40, -4.8790e-40, -3.4137e-40,
 2.2577e-40, -5.7183e-40, -8.6861e-41,
 5.7021e-40, -3.2349e-40,  1.9655e-40,
 9.1180e-02,  5.6665e-02, -6.5437e-04,
 1.1759e-01,  2.7517e-01,  1.9143e-01,
 9.7905e-02,  6.6707e-02,  8.6535e-02,
 8.8717e-03,  3.0913e-02,  6.6909e-03,
-8.1791e-02, -4.7883e-01,  7.4920e-02,
 4.5843e-01, -1.0410e-01,  1.6655e-01,
-4.7094e-03,  3.4769e-02, -1.3291e-02,
-8.5570e-03, -4.0038e-01,  1.8418e-01,
-1.4696e-01,  3.2279e-01,  2.5712e-02,
-2.6207e-01, -4.6150e-02, -6.4099e-02,
-3.2623e-01, -1.8984e-01, -5.7891e-02,
-2.2088e-01, -4.2042e-02, -2.5307e-02,
 1.0260e-40,  5.0443e-40,  7.5150e-41,
 1.4402e-40, -5.1952e-40, -5.3810e-40,
 6.2240e-40,  1.8661e-40, -8.2983e-41,
 7.1850e-02,  4.8770e-02, -1.5081e-02,
 4.8072e-01,  2.5477e-01,  3.8197e-02,
 2.6011e-01,  2.4610e-01, -3.6167e-02,
 3.8901e-40,  1.6760e-41,  2.8471e-40,
 3.1983e-40,  1.2460e-40, -4.3961e-40,
 3.9187e-40,  2.7818e-40, -9.1501e-41,
-2.3320e-40, -1.9998e-40, -2.8132e-40,
-2.9552e-40, -3.9643e-40, -5.1375e-40,
-1.6686e-40, -5.3138e-40, -2.6988e-40,
 2.5623e-02,  2.6942e-02,  2.4342e-02,
-9.9084e-02,  5.2974e-01, -6.7983e-02,
-2.2454e-01,  1.1507e-01,  2.0364e-02,
 3.4852e-01, -3.1091e-01,  8.1154e-02,
-3.2205e-01,  1.7103e-01,  2.4162e-01,
-2.6892e-03,  2.4142e-02,  5.5540e-02,
-4.5753e-02, -5.0097e-01,  1.7503e-01,
 1.4058e-01,  1.1311e-01,  1.5945e-01,
-5.3975e-02,  5.2326e-02, -6.2382e-02,
 9.4114e-02, -5.6812e-01, -1.2081e-01,
-8.5809e-02, -9.8661e-03, -2.3064e-02,
-1.6453e-03, -1.8328e-02,  2.4282e-03,
 1.5943e-40,  4.6894e-40, -6.2730e-40,
 3.8054e-40, -3.7914e-41, -1.4429e-40,
 1.6925e-40,  5.1566e-41, -1.7909e-40,
-3.7920e-02,  2.4698e-01,  5.0019e-02,
-1.4246e-02,  2.8739e-01, -5.4704e-02,
 7.9436e-02, -2.7838e-02, -3.4191e-02,
-3.3565e-40,  2.1368e-40,  6.7346e-42,
 5.6681e-40, -5.5776e-40, -2.7705e-40,
-2.2966e-40,  1.1692e-40, -2.5187e-40,
 4.4806e-40, -4.8424e-40, -9.1436e-41,
-4.3250e-40, -2.0721e-40, -2.0050e-40,
-5.1061e-40,  2.6405e-40, -3.0913e-40,
-1.2078e-01,  3.1948e-01,  1.0082e-02,
-1.0781e-02,  8.0720e-02, -4.6330e-02,
-1.8084e-02, -2.2846e-02, -5.5861e-03,
-3.2400e-02, -1.7329e-01, -2.7995e-02,
-5.3680e-02,  4.1310e-01, -9.4691e-02,
 7.6938e-02, -4.9596e-02,  1.9649e-01,
 3.2594e-02,  1.1544e-01, -1.8501e-02,
 7.0248e-02, -6.9838e-02, -5.4278e-02,
-2.9317e-02, -1.4890e-01,  7.8661e-02,
 3.7685e-02,  5.9594e-02,  8.9527e-02,
 2.2957e-01, -2.9681e-01, -1.6329e-01,
-1.3206e-01, -4.3808e-02,  3.8854e-02,
 1.7529e-40, -3.8429e-41,  1.4443e-40,
-4.0829e-40, -2.5643e-40, -5.4821e-40,
 1.6827e-40, -1.1628e-40,  2.2441e-40,
 5.2451e-02,  1.0179e-01,  4.8487e-02,
-2.1020e-01, -4.4345e-01, -8.7642e-02,
 7.0958e-02,  1.9934e-01, -2.1090e-02,
-3.0795e-41,  2.7921e-40,  2.8491e-40,
-2.1154e-40,  9.8876e-41, -8.8824e-41,
 2.6552e-40,  2.5767e-40, -3.8369e-40,
 6.1348e-40, -3.4170e-40, -1.7109e-40,
-3.3080e-40,  5.4199e-41, -1.7512e-40,
 1.8363e-40, -4.4080e-40, -2.5508e-40,
-4.0716e-02, -2.8531e-01,  3.9981e-02,
 2.2278e-02,  5.6661e-01, -8.3890e-02,
-7.7331e-02, -9.3843e-02,  1.5584e-02
}
,)"
R"(
{
-3.6751e-40, -5.4562e-41,  6.1860e-40,
 8.9003e-41,  5.5262e-40,  3.9537e-40,
-2.1258e-42, -3.1069e-40, -7.6225e-41,
-1.2220e-02, -8.6886e-02,  1.0714e-02,
 1.1656e-02, -7.3635e-02,  5.9427e-02,
 4.8518e-03,  1.3543e-01,  1.4668e-02,
-1.7505e-02, -2.0691e-02, -1.4507e-02,
 2.6157e-02,  7.4109e-02,  1.2822e-02,
-1.9737e-02, -4.9281e-02,  8.5962e-03,
 5.6236e-40,  2.4616e-40,  1.6384e-40,
-3.9469e-40, -1.7094e-40,  1.9285e-40,
-1.3634e-40, -1.5785e-40,  6.4184e-41,
-1.2752e-02,  2.3150e-02, -5.3355e-03,
-5.9667e-02, -3.9580e-01, -7.0033e-02,
-2.2612e-02,  1.9176e-02,  1.0588e-02,
 8.0027e-04,  3.2242e-01, -2.2566e-02,
 8.7850e-03, -2.4025e-01,  4.6123e-02,
-1.9038e-02, -8.5750e-03, -4.8153e-03,
-1.3049e-03, -5.7771e-03,  9.6437e-03,
 3.2477e-02,  2.4482e-01,  4.0580e-02,
 1.3194e-02, -4.6602e-01, -6.6163e-02,
-1.0647e-01,  7.3328e-02,  2.5871e-02,
-7.0883e-02, -9.2725e-02, -1.5185e-02,
 1.1804e-02,  1.7784e-03, -4.4099e-03,
-4.9226e-40, -1.3081e-40, -3.5969e-40,
 4.3539e-40, -2.9631e-40,  2.3531e-41,
 5.6191e-40,  6.1545e-41, -1.1112e-40,
-1.1880e-02, -3.1884e-02, -2.0850e-02,
-6.8633e-03,  1.6422e-01,  1.0281e+00,
 3.5887e-03,  2.1180e-01, -1.0094e-01,
-1.5103e-02, -4.9074e-02, -1.7702e-02,
 7.2119e-02,  3.3199e-02, -9.7082e-04,
 5.5383e-02,  1.0343e-01,  2.5156e-02,
 2.9049e-40, -1.6397e-40, -8.8848e-41,
-6.2827e-40,  8.1281e-41,  5.2909e-40,
-4.1132e-40,  1.5751e-40,  1.5400e-40,
-7.3765e-02, -4.9723e-02,  4.9357e-02,
-2.4207e-02, -1.0291e-01, -1.4001e-03,
-1.2751e-02,  4.2805e-03,  1.8934e-03,
 2.6862e-02,  1.1634e-01,  4.5666e-02,
-4.7351e-03, -4.1593e-01,  3.6082e-02,
 1.1446e-02, -5.2026e-03,  1.8672e-02,
-7.0960e-04, -6.7877e-03,  9.6674e-03,
-4.9952e-03,  8.8664e-02, -2.7707e-02,
 8.5309e-02,  5.5513e-02, -7.6230e-02,
 3.6354e-02,  9.7794e-02,  1.1687e-02,
 2.6847e-02,  3.2565e-01, -8.7710e-03,
-2.0372e-02, -1.9090e-02, -3.2566e-03,
-5.5592e-40,  7.4408e-41,  3.5576e-40,
 2.7758e-40,  4.5458e-41, -6.2347e-40,
 9.9739e-41, -1.6078e-40, -5.2900e-40,
 1.1500e-02, -3.0675e-01, -3.0079e-02,
 1.5080e-02, -2.4292e-01,  1.2736e-01,
-1.9513e-02, -1.9376e-02, -8.5960e-02,
-1.0241e-01, -2.1312e-02, -3.1999e-02,
-6.3598e-02,  1.5187e-01,  1.2279e-01,
 1.5695e-03,  1.1376e-01,  5.2648e-03,
 2.6415e-40,  3.0508e-40,  3.6407e-41,
-1.4403e-40,  2.8942e-40, -1.0089e-40,
 2.2362e-41,  1.9843e-40, -1.5509e-40,
 1.3269e-01, -3.1031e-01, -4.4091e-02,
 4.6385e-03,  2.1411e-02,  5.7141e-02,
 2.0724e-02, -3.5406e-02,  2.5717e-03,
-5.5922e-02,  7.1404e-01, -2.9852e-02,
 1.3041e-02,  3.9373e-02, -2.4515e-01,
 4.4278e-03,  2.1557e-02, -8.4940e-03,
 1.3677e-02, -3.5183e-02,  1.2391e-02,
-9.2405e-02,  2.9650e-01,  6.9695e-02,
-3.3125e-02,  3.4700e-01,  1.4552e-01,
 2.7357e-02,  5.2133e-01, -5.7571e-02,
 2.7580e-02,  1.0381e-01,  1.3678e-02,
 4.9260e-03, -4.4419e-02,  7.0651e-04,
 2.9472e-40, -5.2892e-40, -3.6567e-40,
 4.9403e-40, -6.2132e-40, -6.2920e-40,
-1.5156e-40, -3.6134e-40,  5.2432e-40,
-5.0427e-03, -2.8247e-03, -5.3734e-02,
-1.5918e-02,  1.8325e-01, -1.7834e-01,
-5.1774e-03,  8.0009e-02,  5.6296e-03,
 3.1480e-02,  2.0665e-02,  2.7806e-04,
 7.3085e-02,  7.7660e-01,  1.1979e-01,
 1.9979e-02,  1.6629e-01,  2.3216e-02,
-5.9701e-40,  9.5583e-41,  1.8231e-40,
-3.3216e-40, -4.1253e-40, -3.3326e-40,
 1.7131e-40,  2.9588e-40, -2.2520e-40,
-1.3337e-01, -4.2777e-01, -1.3569e-01,
 2.9915e-02, -2.7016e-01, -3.7454e-03,
-1.3574e-02, -3.6298e-02, -1.6571e-02,
 4.2530e-02, -4.2299e-02,  1.4320e-01,
 1.4371e-02, -1.1289e-01, -3.8829e-02,
 5.1689e-03,  1.5804e-02,  1.6125e-03,
-3.4601e-03, -7.2087e-03, -5.5514e-04,
 4.4568e-02,  1.3621e-01, -4.3811e-02,
 1.1350e-02, -2.8417e-01,  3.1553e-02,
-7.8854e-02, -2.0316e-01,  7.7746e-03,
-1.1437e-02,  2.1557e-01, -1.9479e-02,
-1.3511e-02, -2.0339e-02, -1.0276e-02,
-8.8977e-41,  5.9533e-40, -3.1413e-40,
-3.1892e-40,  5.5204e-40, -5.0634e-40,
-2.4932e-41,  4.3474e-41,  6.2961e-40,
 4.7864e-03,  5.7125e-02, -1.5468e-02,
-3.9614e-03, -2.9042e-02,  2.8347e-01,
-1.0133e-02,  8.2745e-02, -1.0450e-01,
 5.9537e-03,  1.4050e-02,  1.9802e-04,
 2.4964e-02,  1.3077e-01, -4.7314e-02,
 6.2744e-03, -1.9068e-01,  5.2593e-02,
-2.0550e-40, -2.4231e-40,  3.3927e-40,
-3.9609e-41,  2.2262e-40,  1.8866e-40,
 2.0788e-40, -1.8012e-40, -1.9375e-40,
-4.7530e-03, -1.2315e-01,  8.2373e-03,
-9.2412e-02,  1.7156e-01,  1.1176e-02,
-1.4081e-02,  1.4694e-02, -1.9475e-02,
-1.5269e-02, -3.8430e-02, -7.4717e-02,
 3.3361e-02, -1.1956e-01,  4.2304e-01,
-2.9924e-03, -3.3035e-02, -3.6560e-02,
-1.2386e-02,  6.3762e-03, -3.7047e-02,
 1.3839e-02, -3.6358e-02,  4.3609e-02,
-8.3692e-03,  4.5794e-01, -3.0761e-01,
 2.2287e-02,  2.5360e-02, -6.1253e-03,
-1.8992e-02, -4.0078e-01,  7.3821e-02,
 5.6517e-03,  4.2348e-02, -2.5642e-02,
 5.5659e-40, -6.1219e-40,  4.1493e-40,
 5.7719e-42, -3.7181e-40, -3.3260e-40,
-4.8241e-41,  5.2207e-40, -1.2199e-40,
-1.2074e-02,  1.7647e-01,  1.1882e-02,
 6.4764e-03, -2.3742e-01, -1.8033e-01,
 2.5866e-02,  6.5985e-02,  3.7191e-02,
 5.1047e-02, -3.0457e-02,  1.2531e-02,
-1.3252e-01,  1.2593e-01, -6.3717e-02,
 4.0794e-02, -1.4786e-02,  1.7139e-02,
 2.4343e-40, -1.7451e-40,  2.0169e-40,
-5.5166e-40,  2.4201e-40, -2.5701e-40,
 2.9947e-40,  2.9321e-40, -1.6015e-40,
-3.6598e-02, -1.8520e-03, -1.6999e-01,
-8.6806e-02, -7.7266e-02, -9.6042e-02,
-2.1342e-02,  2.5793e-02, -7.2541e-03,
 3.0667e-02, -2.6287e-01,  3.0592e-02,
-4.5559e-02, -1.4716e-01,  2.0932e-01,
-5.8472e-03, -1.0023e-02,  1.2134e-02,
-1.3284e-02,  2.0538e-02, -5.4476e-04,
 5.8096e-02, -1.4790e-02, -2.0158e-02,
-3.9654e-02, -2.2069e-01, -1.5089e-01,
-1.8966e-01, -1.6834e-01,  9.8934e-02,
 8.2326e-02,  7.5585e-02, -1.7188e-02,
-1.4985e-02,  2.1823e-02, -7.7015e-03,
 1.8353e-40,  4.8298e-40, -2.0568e-40,
-3.7196e-40, -5.7237e-40,  1.0648e-40,
 9.4960e-41,  3.0411e-40,  1.3294e-40,
-1.4884e-02,  4.9767e-02, -3.0288e-02,
 8.9874e-03, -1.0290e-01,  3.1344e-01,
 5.9735e-03, -2.0813e-01, -6.6145e-03,
 1.6592e-02,  3.0529e-05, -1.0180e-02,
-4.8683e-02,  1.4025e-01,  2.9237e-02,
-2.3334e-02, -9.6638e-02, -1.0268e-02,
-4.9497e-41, -5.6377e-40, -2.0142e-40,
 2.1230e-40,  1.6067e-40,  3.4830e-40,
-4.9031e-40, -3.0290e-40, -2.9060e-40,
 3.4053e-02, -8.9560e-02, -4.4479e-02,
 4.2128e-02,  6.9253e-02, -7.1096e-03,
 4.2358e-02, -1.7215e-02,  9.0389e-03,
 1.8129e-02, -1.4785e-01,  1.1267e-01,
-7.1637e-02,  5.5595e-01, -1.0569e-02,
 1.8481e-02, -4.7556e-02, -1.1185e-02,
-1.1766e-02, -8.5959e-03, -3.0046e-02,
-2.1081e-03,  1.1518e-01, -8.4419e-02,
-7.5829e-02,  1.8199e-01, -9.7726e-03,
 3.6473e-02,  1.8761e-01,  4.9495e-03,
-6.9640e-02, -2.8775e-01,  3.6149e-02,
 9.6345e-04,  1.3967e-02, -6.0015e-03,
 2.9861e-40,  3.9190e-40,  5.3741e-40,
 3.8059e-40,  4.7113e-40,  5.9498e-40,
-5.0640e-40, -4.1610e-40,  6.2009e-40,
-2.3464e-03, -7.3888e-02,  3.4701e-02,
-5.2257e-04,  3.8444e-02, -5.3735e-01,
-1.7970e-03,  9.0298e-02,  5.3151e-02,
-2.6033e-02,  1.2973e-02,  4.9147e-03,
 2.3005e-02,  1.7045e-01,  2.4715e-02,
 2.7981e-02, -8.4662e-02, -9.4778e-03,
 5.3019e-40, -2.1800e-40,  1.5281e-40,
-1.0282e-40,  1.8040e-41,  1.3929e-40,
-5.9679e-40, -5.2958e-40,  1.4429e-40,
 3.4325e-02, -1.7240e-01, -4.9645e-02,
-2.4341e-02,  5.2652e-02, -1.1188e-02,
-3.6336e-03,  4.2148e-04,  3.3086e-03,
 5.5059e-03,  1.7744e-01, -2.8681e-02,
-3.4868e-03, -1.4569e-01,  1.6508e-02,
 4.6766e-03, -1.7963e-02, -2.6397e-03,
 4.3618e-03, -4.2793e-03, -4.7820e-04,
-4.2795e-02,  2.0070e-01,  3.8402e-02,
 5.0586e-02,  2.1910e-01, -3.4381e-02,
 5.7625e-02,  4.2314e-01, -1.9732e-02,
 3.4811e-02, -2.3033e-01,  1.1477e-02,
-7.3744e-03,  1.9112e-02,  4.2251e-03
}
};
)"
R"(
__constant float biasL[8][8] = 
{
{
 0.0272, -0.5743, -0.0333, -0.0334,  0.0082, -0.0263, -0.0048, -0.0167
}
,
{
-0.0239, -0.0385,  0.0026,  0.0288, -0.0225,  0.0082, -0.0191, -0.0185
}
,
{
-5.8305e-03, -8.6574e-02,  4.2228e-02, -4.3500e-02, -8.1892e-04, 3.3171e-03, -1.1582e-02, -4.1205e-40
}
,
{
-0.0053,  0.0053, -0.0114, -0.0127, -0.0039, -0.0426,  0.0053, -0.0017
}
,
{
-0.0046, -0.0104, -0.0087, -0.0040,  0.1077,  0.0347, -0.0165,  0.7296
}
,
{
 8.7612e-02,  5.9126e-01,  4.6709e-03, -1.1559e-39,  2.3381e-02, -1.2136e-40, -5.6040e-39,  3.7100e-02
}
,
{
-3.3246e-39, -1.4536e-02, -6.3362e-02,  8.5347e-41,  7.9956e-02, 3.0679e-04, -1.0257e-02, -1.2037e-02
}
,
{
-0.0006,  0.0117,  0.0083,  0.0686, -0.0046,  0.0015, -0.0076,  0.0079
}
};

__constant float kernelsL10[4 * 8] = 
{
 0.4908, -0.0457,
-0.1716, -0.2115,
-0.0015, -0.3152,
 0.3045,  0.0330,
-0.2981,  0.0912,
 0.0122,  0.2281,
 0.3331,  0.2853,
 0.2210,  0.2611,
 0.2364,  0.0792,
 0.2885, -0.7122,
-0.3715,  0.1404,
-0.0260,  0.2144,
 0.2378,  0.1570,
-0.5734,  0.2077,
-0.0851,  0.2771,
 0.0415, -0.1858
};)") + kernelFunction
,
std::string(
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
-6.6326e-02, -2.2316e-01,  4.2471e-02,
 1.7064e-02, -6.8305e-01, -1.5978e-01,
 6.7568e-01,  3.2212e-01,  8.3561e-02,
-4.6649e-01, -6.8789e-02,  5.3455e-01,
-5.0941e-01,  7.0657e-02,  4.5647e-01,
-2.3657e-02,  3.5302e-02, -1.8316e-02,
-2.0316e-01,  4.7021e-02, -2.2313e-01,
 5.3465e-02,  7.0750e-01,  9.1366e-02,
-2.8566e-01, -2.0521e-02, -7.1786e-02,
 4.8186e-02, -9.3429e-02,  2.4493e-03,
 3.4654e-01,  7.2625e-02,  1.6615e-01,
 3.2101e-01,  3.2923e-01, -9.8548e-02,
 1.1916e-02,  2.0413e-01, -1.8920e-02,
 6.0858e-02,  8.3548e-01,  1.4060e-01,
-9.1827e-01, -2.4551e-01, -4.6118e-02,
-5.2737e-02,  4.3151e-01,  1.7027e-01,
 2.6647e-01,  5.5240e-01,  3.4745e-03,
 5.3495e-02, -4.7059e-02, -2.6593e-02,
 1.5691e-01,  4.7332e-01,  2.6651e-03,
 1.7997e-02,  4.1367e-01,  1.3239e-02,
 4.6932e-02,  1.0278e-01,  1.0699e-02,
-3.4319e-02, -7.6373e-01, -9.7022e-02,
-1.4160e-01,  2.9567e-01,  6.6220e-01,
 7.3508e-05,  1.2683e-01, -6.3442e-02
};

__constant float biasL1[8] = 
{
-0.0264, -0.0229, -0.3021, -0.2579, -0.0327, -0.0053, -0.7777,  0.0232
};
)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
-7.8588e-41, -5.0770e-40, -2.3334e-40,
 5.7174e-40,  6.9060e-41,  2.2264e-40,
-4.1631e-40,  4.5667e-40, -1.8115e-40,
-3.1000e-40,  3.1019e-40,  5.5423e-40,
-5.8518e-40,  2.1290e-40, -5.4579e-40,
-3.7753e-40,  3.6029e-40, -1.7875e-40,
 4.2296e-40,  6.5672e-41,  1.4976e-40,
-3.1479e-40, -3.2881e-40, -5.9818e-40,
 3.2053e-40,  3.0821e-40,  5.1321e-40,
-2.6557e-17, -3.8205e-17, -3.7077e-17,
-2.5168e-17, -3.4817e-17, -3.4186e-17,
-1.8056e-17, -2.3105e-17, -2.2581e-17,
 5.9355e-40,  2.4052e-40, -1.0027e-40,
 2.2060e-40,  3.4864e-40, -5.7403e-40,
 4.6936e-40, -3.3951e-40, -4.7715e-40,
-9.7917e-11, -1.0331e-10, -9.6141e-11,
-1.0581e-10, -1.1173e-10, -1.0317e-10,
-1.0192e-10, -1.0681e-10, -9.8738e-11,
-1.0402e-29, -2.3233e-29, -1.7882e-29,
-1.4804e-29, -3.7821e-29, -3.0750e-29,
-1.0448e-29, -2.6740e-29, -2.1676e-29,
 4.2124e-40,  2.5024e-40,  4.5312e-40,
-2.4880e-40,  2.9838e-41, -2.7215e-41,
-2.6347e-40,  1.5950e-40,  9.3734e-41,
-1.4936e-01, -1.0438e-01,  2.9827e-02,
 1.4751e-02, -1.6854e-01, -8.8101e-02,
 4.9228e-02, -3.0744e-02, -1.1512e-01,
-3.4996e-02, -2.5024e-02, -1.8880e-02,
 3.0008e-02,  4.8689e-02, -1.3415e-01,
-9.1698e-03, -1.1019e-02, -5.0655e-02,
-6.6579e-02, -2.6447e-02,  1.9791e-02,
-4.1727e-02,  3.6433e-02,  3.1516e-02,
-5.7619e-02,  2.3401e-02,  3.0785e-02,
-3.3610e-02,  1.2263e-01,  2.4351e-02,
 1.7148e-02,  1.7144e-01,  4.0305e-02,
 8.7902e-03, -7.0077e-02, -1.0688e-01,
 4.7460e-02, -1.4093e-03, -1.5911e-02,
-2.2978e-02,  9.9025e-02,  1.2867e-02,
 3.4704e-02,  1.4672e-01,  7.9188e-02,
-4.4222e-02, -3.9480e-02, -1.9193e-01,
-3.1897e-02,  1.0776e-01, -5.2742e-02,
 8.0377e-02,  2.5764e-01, -9.7330e-02,
-1.1593e-01, -5.3753e-02, -2.8918e-02,
 6.7939e-02,  2.3963e-01,  2.0856e-01,
 2.7964e-02,  2.7781e-01,  2.1859e-01,
-1.5196e-02,  9.6704e-03, -8.0136e-02,
 8.9441e-02,  1.0314e-01, -2.0204e-02,
-3.3970e-02, -1.4562e-02,  3.4723e-02,
 2.3357e-40, -1.4361e-40,  2.0498e-40,
-5.2355e-40, -6.0151e-40, -2.9264e-40,
 1.9715e-41,  5.9793e-41, -1.3675e-40,
 5.3771e-40,  6.5637e-41, -3.8471e-40,
-3.0820e-40, -1.7004e-40, -1.9371e-40,
-5.1159e-40,  7.3244e-41,  3.5861e-41,
 2.8441e-40,  4.5248e-41,  1.9771e-40,
-2.4681e-40,  3.6054e-40,  3.3496e-40,
-6.5048e-42, -1.6001e-40,  4.8243e-41,
-1.0165e-08, -9.9140e-09, -9.6054e-09,
-1.0511e-08, -1.0256e-08, -9.9066e-09,
-1.0521e-08, -1.0320e-08, -9.9896e-09,
 2.6042e-40,  4.2016e-40,  5.3537e-40,
 1.4594e-40,  1.1344e-40,  3.5144e-40,
-2.5736e-37, -1.3591e-39,  2.1029e-40,
-3.1420e-07, -3.0309e-07, -2.9630e-07,
-3.1196e-07, -2.9967e-07, -2.9249e-07,
-3.1296e-07, -3.0086e-07, -2.9332e-07,
-6.1256e-12, -5.9283e-12, -5.6508e-12,
-6.5297e-12, -6.4118e-12, -6.0667e-12,
-6.8382e-12, -6.8547e-12, -6.5225e-12,
-5.0327e-26, -1.0795e-25, -1.8952e-25,
-2.4220e-26, -5.9067e-26, -1.1323e-25,
-2.1499e-27, -5.5342e-27, -1.0333e-26,
 4.5039e-03, -1.3303e-02,  1.6183e-01,
 6.5951e-02, -7.1353e-02,  1.7254e-01,
-1.8671e-03,  1.0593e-01, -3.6872e-02,
 4.9102e-02, -2.4075e-03,  4.8194e-02,
-7.0892e-02, -1.8948e-01, -1.6586e-01,
-2.8102e-02,  2.0870e-02,  5.9228e-02,
 1.2673e-02,  3.3908e-02,  4.8282e-02,
 4.4369e-02,  5.6304e-02,  1.2225e-02,
 4.1855e-02,  1.1990e-01,  6.3799e-02,
-7.3884e-02,  1.4153e-02,  9.5825e-02,
 4.2850e-02, -3.5337e-02,  1.3615e-01,
-2.0900e-01, -2.2835e-02, -8.6987e-02,
-6.7793e-02,  1.3547e-01, -9.9666e-02,
 3.5498e-02,  5.3725e-02,  1.1501e-01,
-1.2238e-01,  3.5354e-02,  7.4216e-02,
-3.5288e-02,  7.0111e-03,  2.4820e-02,
-1.0649e-02,  1.6715e-01,  1.2825e-01,
 3.1145e-02,  1.2097e-01, -1.2073e-02,
-7.0603e-02,  5.5574e-02, -5.0025e-02,
-8.2885e-02,  1.0957e-01,  1.3311e-01,
 2.9147e-02, -1.1849e-02,  8.9953e-02,
-3.2247e-02, -1.0747e-02,  9.1431e-03,
 1.2114e-01, -5.9780e-02,  5.4821e-02,
-5.2592e-02, -6.9082e-02, -7.5981e-02,
-7.8533e-02,  1.3658e-01,  1.0923e-01,
-3.2530e-02, -2.1342e-01, -1.2200e-01,
-1.9196e-02,  1.0450e-01, -8.9044e-02,
-2.0110e-02,  6.1439e-02, -2.7405e-02,
 6.0823e-02, -6.4268e-03, -9.1778e-03,
 6.4877e-02, -6.1227e-02, -5.4466e-02,
 9.6375e-02,  1.7519e-01,  5.0725e-03,
 1.9159e-01,  3.9725e-01,  1.2851e-01,
-6.9197e-02,  4.9372e-02, -3.4221e-02,
 1.1583e-01,  1.3389e-01,  2.9135e-01,
 1.0290e-02,  1.1214e-01,  1.7560e-01,
-1.8048e-02,  8.4782e-02,  4.9925e-02,
-3.8447e-02, -1.3156e-01, -1.1072e-01,
 1.8256e-01,  2.2831e-01, -1.6508e-01,
 4.6781e-02,  1.4913e-01, -8.6956e-02,
 5.1365e-04,  6.7873e-02, -3.4787e-03,
 1.7689e-01,  1.8414e-01,  2.2286e-01,
 1.2571e-01,  1.7687e-01,  1.5949e-01,
 5.9904e-02,  1.6259e-01,  1.4313e-01,
 2.2234e-01,  4.0943e-01,  3.1469e-01,
 1.9799e-01,  4.3052e-01,  3.0510e-01,
 1.2259e-01, -1.0778e-02,  6.2284e-03,
 1.4508e-02, -6.9073e-02,  5.0998e-02,
 5.2962e-02, -1.5291e-01, -1.0491e-02,
-8.6903e-02, -1.0430e-01,  3.0130e-02,
 4.1691e-02, -1.2675e-01, -5.5169e-01,
 8.9644e-02,  3.6910e-02, -1.5459e-01,
 5.3656e-03,  6.7936e-02,  1.0793e-01,
-2.7424e-02, -1.7652e-01, -3.5776e-01,
 2.4593e-02, -5.6237e-01, -5.9038e-01,
-9.4807e-02, -7.5681e-02, -3.6990e-02,
 8.7385e-03, -5.7989e-02, -4.9573e-02,
-7.7422e-02, -1.1899e-01, -7.4023e-02,
 9.1539e-03, -1.1760e-01,  4.6825e-02,
 1.9901e-02, -3.9718e-02,  1.2997e-02,
 4.2209e-02, -5.2119e-02, -1.2255e-01,
 2.4262e-02,  5.3676e-02, -2.4767e-01,
-4.2933e-02, -2.2473e-01, -4.0310e-01,
-3.5160e-02,  1.9858e-01, -1.5943e-01,
 1.3208e-01, -1.0493e-01, -6.7076e-02,
-2.5244e-01,  1.1175e-02,  2.5568e-01,
-3.3867e-01,  3.1953e-02,  5.9426e-01,
 4.0551e-02,  4.4914e-03, -1.9348e-02,
-6.7386e-02, -1.5543e-01, -3.0883e-02,
 8.9177e-02, -4.6432e-02,  6.8227e-02,
 8.7784e-02,  3.6127e-02, -2.0375e-02,
 4.5461e-02, -4.9071e-02,  9.9435e-02,
-2.5700e-01, -2.7706e-01,  6.2776e-02,
-6.9571e-02, -5.7888e-03,  9.3852e-02,
 2.8490e-02, -2.7854e-01,  1.4209e-01,
 1.5373e-02, -4.3503e-02,  9.6895e-02,
 1.1682e-02,  1.5608e-01,  1.5844e-01,
 5.8027e-02,  2.6632e-02, -8.5479e-03,
 1.2836e-01,  2.0714e-01,  1.0228e-01,
 1.4647e-02,  5.7609e-02, -1.6728e-02,
 2.1212e-01,  3.2673e-01,  4.5670e-02,
-6.0844e-02, -1.1768e-01, -1.1233e-01,
 5.0123e-04,  6.3947e-02, -1.8356e-01,
 1.4091e-01, -2.1568e-02,  8.5933e-02,
-3.9406e-02,  8.2921e-02, -1.0601e-01,
 4.1284e-02, -7.3138e-02,  1.7264e-01,
 2.5883e-02,  5.2945e-01,  2.4510e-01,
 2.7291e-03,  4.0173e-02,  7.8221e-03,
-3.5795e-02, -4.8631e-03, -2.2715e-01,
 1.2330e-01,  7.1739e-01, -4.1725e-01,
 7.5106e-02,  2.5267e-02, -2.8655e-01,
-7.8731e-02, -7.5747e-03, -5.5601e-02,
 7.9764e-02,  1.0524e-01,  8.6742e-03,
 2.1791e-02,  3.7304e-02, -1.1534e-01,
-1.2011e-01, -7.5160e-02,  1.3737e-02,
-2.9470e-01,  2.6613e-01, -2.3740e-02,
 1.2957e-01,  1.4752e-01, -9.3655e-02,
 2.9828e-02,  2.0664e-01,  1.9731e-02,
-8.0378e-02, -3.9481e-01, -1.5395e-01,
-5.7944e-02, -8.6343e-02, -5.4324e-02,
 7.1664e-02,  1.5294e-01, -1.2112e-02,
 2.1023e-02,  1.1945e-01, -7.2998e-02,
-1.1693e-02, -1.8818e-01, -9.8693e-02,
-6.7017e-02,  6.9767e-02, -5.0268e-02,
-9.1106e-03,  2.4267e-01,  6.0277e-02,
 3.5269e-02,  7.7376e-02,  1.6642e-02,
-5.2600e-02, -1.8864e-01, -1.1195e-01,
 3.2119e-01, -9.7913e-02,  1.4734e-01,
 8.6988e-02, -5.3563e-03, -2.6136e-03,
-9.1528e-03,  2.8186e-01, -1.5933e-01,
 4.8499e-02,  4.5189e-01, -1.6399e-01,
 5.8164e-02,  6.3251e-02, -2.8738e-02,
 2.0424e-01, -7.2819e-02,  2.1903e-02,
-3.5630e-01,  1.3171e-01, -7.6749e-02,
 3.8848e-02,  1.7902e-01, -1.1902e-01,
-4.4221e-02,  1.5032e-02,  2.9078e-02,
-1.9738e-01, -1.4878e-02,  1.3315e-02,
 1.3956e-02,  1.2856e-01,  7.0688e-02,
 2.0933e-01,  1.7286e-01,  6.7601e-02,
 5.5136e-01,  4.6866e-01,  1.8402e-01,
 2.2362e-01,  2.4124e-01,  1.3167e-01
}
,)"
R"(
{
-5.2308e-12, -5.4024e-12, -5.0039e-12,
-5.4553e-12, -5.6928e-12, -5.2812e-12,
-5.0230e-12, -5.2150e-12, -4.9133e-12,
 5.7994e-02,  1.0051e-01, -1.0618e-01,
 6.8090e-02,  1.2789e-01,  1.1380e-01,
-1.5882e-01,  8.2323e-03, -9.1424e-02,
 2.0132e-07,  2.0907e-07,  2.1344e-07,
 2.1179e-07,  2.2018e-07,  2.2381e-07,
 2.1095e-07,  2.1920e-07,  2.2150e-07,
 2.9336e-02,  5.4427e-02, -1.2082e-01,
 5.8399e-02,  2.2261e-01,  1.1165e-01,
-9.6098e-02,  8.3175e-02, -6.5909e-02,
 1.2007e-01,  1.9776e-01,  7.7464e-02,
 6.7018e-02,  3.6536e-01,  1.3796e-01,
 6.0724e-02,  4.6161e-02,  2.3740e-01,
-2.1117e-02, -2.0200e-02,  9.3703e-02,
-4.6932e-02, -1.5910e-01,  8.8094e-02,
-5.6641e-02, -1.7146e-01, -1.0502e-01,
-2.5624e-01,  1.6049e-01, -3.3267e-02,
-2.3248e-01,  5.4036e-01,  1.0027e-01,
-2.1680e-01, -7.0096e-03, -1.0692e-01,
-4.8357e-02,  2.5107e-01,  4.8323e-02,
 9.7245e-02,  5.5015e-01, -3.4641e-01,
 1.2458e-02, -1.3626e-01, -4.1992e-01,
-2.1359e-40, -1.4250e-40, -4.7123e-40,
-5.9433e-41,  1.9903e-41, -1.7701e-40,
-5.9941e-40, -5.8562e-40, -5.0226e-40,
-2.6581e-40,  1.3006e-40, -1.4201e-40,
 5.4264e-40,  2.3848e-40,  5.6412e-40,
-2.6378e-41, -5.7132e-40, -4.1343e-40,
-3.2848e-22, -3.6697e-22, -3.4147e-22,
-3.5780e-22, -3.9435e-22, -3.5989e-22,
-3.1212e-22, -3.4305e-22, -3.0670e-22,
-1.1749e-08, -1.1602e-08, -1.1494e-08,
-1.2125e-08, -1.1918e-08, -1.1718e-08,
-1.1779e-08, -1.1623e-08, -1.1559e-08,
-5.0237e-07, -4.9179e-07, -4.6744e-07,
-5.1967e-07, -5.0826e-07, -4.8421e-07,
-5.0226e-07, -4.9668e-07, -4.8019e-07,
 5.6433e-41, -3.0514e-40, -5.4526e-40,
 1.1125e-41,  2.9485e-40,  5.5282e-40,
 3.0229e-40,  1.5915e-40,  5.3759e-40,
-6.1144e-27, -9.2380e-26, -2.4302e-25,
-9.3834e-25, -1.0289e-23, -1.9513e-23,
-4.3746e-24, -4.4359e-23, -7.0505e-23,
-8.1604e-36, -3.2928e-37, -2.2994e-40,
-3.9543e-37, -9.9513e-39,  7.4616e-41,
-4.0044e-39,  4.4392e-40,  4.8856e-40,
-3.3447e-40, -3.9935e-40,  2.4649e-40,
 2.0207e-40, -3.0245e-40, -7.1986e-41,
 6.2938e-40, -3.6922e-40,  1.5296e-40,
-6.4982e-41,  5.0849e-41,  5.7873e-40,
 1.4327e-40, -4.2163e-40,  1.3807e-40,
 2.8569e-40,  1.9139e-40,  3.2985e-40,
-5.4410e-40,  2.3070e-40,  2.1690e-40,
-1.5964e-40, -2.2781e-40,  5.6766e-40,
 2.2533e-42, -2.5532e-40, -5.5822e-40,
 5.7249e-40,  5.3555e-40, -4.9107e-41,
 1.7538e-40, -1.2312e-40,  5.0077e-40,
 6.1500e-40,  1.9980e-40,  6.2953e-40,
-7.5314e-23, -9.4299e-23, -7.1342e-23,
-8.5139e-23, -1.1237e-22, -9.0478e-23,
-6.2038e-23, -8.5180e-23, -7.3015e-23,
 5.0613e-40,  1.5224e-40, -1.8977e-40,
 2.4108e-41, -5.1771e-40,  6.2317e-40,
 1.0465e-40,  2.8816e-41,  6.2500e-40,
 3.5727e-40,  4.2717e-40, -3.5900e-40,
-4.4831e-40,  3.4260e-40, -4.8293e-40,
-2.4133e-40,  3.1140e-40, -2.0777e-40,
-2.2906e-41,  3.5923e-40, -4.4443e-40,
-4.6615e-40, -2.1123e-40,  4.5700e-40,
-4.6360e-40, -3.6052e-40, -3.4319e-40,
-3.6575e-40, -3.5707e-40, -3.0530e-41,
 4.2531e-40, -1.2255e-40, -3.9607e-40,
 3.5903e-40, -5.4630e-40, -3.1460e-40,
 2.8820e-40,  4.9460e-40,  6.1461e-40,
 8.9118e-41, -4.6579e-40, -2.4172e-40,
-5.5474e-40, -8.1848e-41, -1.6910e-40,
-1.6272e-25, -1.8802e-25, -1.7229e-25,
-1.7850e-25, -2.0338e-25, -1.8235e-25,
-1.4715e-25, -1.6733e-25, -1.4681e-25,
-5.5471e-09, -5.6862e-09, -5.7043e-09,
-5.8727e-09, -5.9823e-09, -5.8983e-09,
-5.8040e-09, -5.8670e-09, -5.7388e-09,
-9.7253e-07, -9.7248e-07, -9.4623e-07,
-1.0149e-06, -1.0042e-06, -9.6709e-07,
-1.0139e-06, -9.9930e-07, -9.5295e-07,
-4.5042e-40,  2.6725e-40,  2.3181e-40,
-4.6274e-41, -1.1799e-40,  5.0685e-40,
-1.0765e-40,  3.3322e-40, -6.1905e-40,
-1.3653e-34, -3.4690e-33, -1.1578e-32,
-1.4444e-31, -2.1995e-30, -4.8668e-30,
-1.2965e-30, -2.0189e-29, -3.3962e-29,
-2.5057e-40,  7.2876e-41,  4.5731e-41,
-1.6525e-40,  5.0987e-40, -5.4683e-40,
 8.1836e-41,  6.2722e-40, -3.1057e-40,
 4.0987e-40,  3.5941e-40,  5.1680e-40,
 5.5563e-40,  3.1011e-40,  4.7068e-40,
 1.0426e-40, -1.0803e-40,  4.4867e-40,
-4.9675e-03,  1.5412e-01, -4.1930e-03,
-6.1089e-02,  2.0405e-01,  1.9587e-01,
 3.8772e-02,  1.6894e-01, -2.6163e-02,
 1.0839e-30,  1.8608e-30,  1.1386e-30,
 1.4863e-29,  1.9422e-29,  1.1639e-29,
 1.7504e-29,  2.2177e-29,  1.3629e-29,
 6.4484e-02,  6.6296e-02,  2.2838e-01,
-1.0213e-01,  7.5883e-02, -1.7531e-01,
-1.4869e-01,  1.0736e-01,  1.4129e-01,
-2.8235e-02, -2.9232e-02, -9.3912e-02,
 5.1317e-02,  9.0256e-02, -2.4669e-02,
-3.2465e-02,  5.8099e-02,  9.8402e-02,
-2.3135e-01, -1.3786e-01,  2.8581e-01,
-3.2410e-01, -2.6623e-01,  6.1583e-02,
 1.8696e-01,  4.7251e-02, -2.3520e-01,
 2.5630e-02, -1.2358e-01, -1.5735e-01,
-1.2198e-01,  5.1970e-01,  1.9976e-01,
-1.2515e-01,  9.8768e-02,  5.8917e-02,
-3.8569e-02, -9.2729e-02, -1.8982e-01,
 1.1378e-01,  5.7195e-01, -1.8265e-01,
-3.5724e-02, -2.1379e-01, -2.2129e-01,
-5.1198e-40, -3.4709e-40,  6.2940e-40,
-2.2134e-41, -3.6133e-40, -2.7075e-40,
-5.9664e-40, -2.3937e-40,  3.0876e-40,
 9.1814e-41,  9.5898e-41, -3.1892e-40,
 3.1093e-40,  2.7935e-40,  1.7966e-40,
-2.3967e-40,  4.0806e-40,  6.2012e-40,
 5.3771e-41,  6.1000e-40, -4.6695e-40,
 5.9474e-41, -4.9675e-40,  5.7403e-41,
 4.7091e-40, -5.0751e-41,  3.9864e-41,
-9.7756e-41,  2.7978e-40, -5.0791e-40,
-3.4321e-40, -7.0774e-41, -5.2651e-40,
 2.8034e-40, -3.3452e-40,  1.9535e-40,
-6.2300e-40, -1.8372e-40, -1.9038e-40,
-5.6564e-40, -6.1257e-40, -1.0338e-40,
-1.7191e-41, -1.2843e-41,  5.0707e-40,
-4.4587e-40,  2.7128e-40, -1.4155e-40,
-5.7475e-40, -3.4612e-40, -4.7424e-40,
 1.7235e-40, -6.0028e-40, -1.6342e-40,
-5.1072e-40, -2.4721e-40, -2.8477e-41,
 2.6598e-40, -4.4078e-40,  4.1763e-40,
-3.3947e-40, -5.5626e-40,  4.9713e-40,
 2.1733e-40, -2.9024e-40, -4.5514e-42,
-3.4873e-40, -1.0737e-40, -1.4297e-40,
 2.8514e-40,  2.6283e-40,  2.2827e-40,
 3.8908e-40, -4.2140e-40,  6.1433e-40,
-4.7825e-40, -3.0140e-40, -5.9563e-40,
 1.5280e-40,  2.6156e-40,  5.0361e-40,
 1.9497e-01,  2.3140e-01, -3.5244e-02,
 1.6876e-01, -1.7646e-02, -2.0413e-01,
 9.8052e-02, -6.7906e-02, -3.9834e-02,
-5.9252e-15, -6.7431e-15, -8.1865e-15,
-5.7350e-15, -6.6893e-15, -8.9833e-15,
-8.4106e-15, -1.0631e-14, -1.5948e-14,
 8.9389e-02,  6.6460e-02,  6.8477e-02,
 6.1099e-03, -8.7536e-02,  1.1792e-01,
-1.0079e-01,  1.5293e-01,  4.3945e-02,
 1.0168e-01,  1.0281e-01, -7.9173e-02,
 2.0855e-01,  1.7537e-01, -7.1000e-02,
-1.4157e-01, -3.8478e-02, -2.7478e-01,
 2.2156e-01, -6.4262e-02, -7.2841e-02,
-3.2334e-01,  6.5591e-02,  1.1163e-01,
 7.2151e-02, -1.6943e-01,  5.9049e-02,
-1.4813e-01, -2.0904e-01, -8.8010e-02,
-2.7215e-01,  5.7668e-01,  1.7618e-02,
-7.1365e-02,  1.2976e-01, -1.0169e-01,
-8.9229e-02,  3.3971e-02,  1.8295e-01,
 1.7204e-01,  3.8082e-01,  3.7415e-02,
 5.9309e-02, -4.9550e-04,  5.1555e-01,
-5.1006e-18, -5.6038e-18, -5.8724e-18,
-5.8910e-18, -5.8379e-18, -5.6311e-18,
-5.2596e-18, -5.1835e-18, -4.6300e-18,
 6.4067e-02,  1.8889e-02, -1.0634e-01,
 1.7316e-04,  1.9935e-01, -1.1854e-02,
-9.3669e-02, -1.1924e-01, -1.8981e-02,
 1.7465e-08,  1.7340e-08,  1.7565e-08,
 1.8234e-08,  1.8008e-08,  1.8017e-08,
 1.9226e-08,  1.8956e-08,  1.8651e-08,
-1.7294e-01, -1.2200e-01, -4.9577e-02,
-3.5087e-02, -1.2526e-01,  9.3445e-03,
-7.4374e-02, -1.1350e-01,  2.7510e-03,
 8.5153e-02,  4.2080e-02, -5.0111e-02,
 1.2845e-01,  1.9630e-01,  1.0542e-01,
-1.0095e-01,  6.2631e-02,  8.8734e-02,
 3.4836e-01,  5.4389e-01, -2.2360e-01,
 5.1721e-01,  5.7094e-01, -6.7491e-02,
-3.5972e-02,  1.0590e-01, -2.2984e-01,
-1.5483e-01, -5.1271e-03,  4.9780e-02,
-1.3184e-01,  2.8028e-01, -1.1427e-02,
-3.4093e-02, -6.7622e-02, -1.2359e-02,
 1.3184e-02,  1.2125e-01, -1.2502e-02,
 9.2730e-02, -6.5974e-02, -1.6519e-01,
 1.9546e-01, -1.5188e-01, -8.1752e-02
}
,)"
R"(
{
-3.4905e-04, -3.5739e-04, -3.2920e-04,
-3.8506e-04, -3.9121e-04, -3.5635e-04,
-3.7303e-04, -3.7698e-04, -3.4190e-04,
 2.8622e-41, -1.2033e-41,  1.2609e-40,
-4.9379e-40, -5.1047e-40,  5.5085e-41,
-4.7002e-40, -5.0136e-40, -4.5629e-40,
-5.1095e-40,  1.8741e-40,  1.8435e-40,
 4.1851e-40, -8.9558e-41, -9.6681e-41,
-1.8244e-40,  2.7992e-40,  1.8116e-40,
 2.8655e-40, -3.0193e-40,  2.2293e-40,
 1.6805e-40,  3.3049e-40,  6.9542e-41,
-3.3329e-40,  4.2212e-40, -1.3453e-40,
-8.4502e-15, -1.1099e-14, -9.4174e-15,
-9.8778e-15, -1.1768e-14, -9.4875e-15,
-6.7805e-15, -7.4561e-15, -5.8023e-15,
 6.0452e-40,  6.9262e-41,  2.9300e-40,
-6.1511e-40, -4.1269e-40,  4.4012e-40,
 1.3340e-42, -2.9020e-40, -4.5529e-40,
-1.2289e-22, -1.3972e-21, -5.5694e-21,
-1.7854e-21, -1.7743e-20, -5.6749e-20,
-6.8510e-21, -6.2353e-20, -1.6203e-19,
-5.0003e-07, -5.1950e-07, -4.7654e-07,
-5.5510e-07, -5.7995e-07, -5.2753e-07,
-5.3262e-07, -5.5802e-07, -5.0971e-07,
-1.4922e-02, -1.1926e-01, -1.9067e-02,
-2.6298e-03,  2.1756e-01,  3.0148e-02,
 1.4372e-01,  3.5066e-02, -1.0184e-02,
-4.1698e-12, -4.8798e-12, -6.4033e-12,
-2.3169e-12, -2.7879e-12, -3.7276e-12,
-1.6177e-12, -2.0021e-12, -2.6440e-12,
-5.9514e-40, -4.4339e-40, -3.0315e-40,
 3.5756e-40,  2.5390e-40, -1.2253e-40,
 2.1417e-40,  4.0569e-40,  5.3962e-40,
-5.5825e-13, -6.8528e-13, -9.3486e-13,
-2.9163e-13, -3.6959e-13, -5.1183e-13,
-1.8703e-13, -2.4740e-13, -3.4019e-13,
-2.7137e-01, -4.5025e-01,  2.6405e-02,
-7.9580e-02,  5.0698e-01, -7.8794e-02,
-3.7540e-02, -7.1115e-03, -3.9741e-01,
-5.9910e-40, -5.5101e-40,  3.1274e-41,
-6.9384e-41, -4.9294e-40, -1.0818e-40,
-3.5484e-40, -4.7965e-41, -5.2508e-41,
 4.1917e-01, -1.6207e-02, -6.8506e-02,
-2.7060e-02,  5.6162e-01,  1.6696e-01,
-1.7677e-03,  1.8842e-01, -6.0493e-02,
-3.0696e-01, -1.7293e-01, -8.7143e-02,
-1.6740e-01,  1.8861e-02, -1.7112e-01,
 8.6594e-02,  3.0025e-01, -7.6141e-02,
 1.1317e-02,  1.0678e-01, -5.1283e-02,
-1.2872e-01,  4.2580e-01,  4.9678e-02,
-2.8372e-01, -1.3479e-01, -7.3813e-02,
-1.7038e-15, -1.1156e-15, -7.3385e-16,
-2.6350e-15, -1.6234e-15, -1.0598e-15,
-7.7860e-15, -4.6981e-15, -3.0030e-15,
-3.0246e-40, -4.1596e-40,  2.9013e-40,
 8.5195e-41, -2.2396e-40, -2.0322e-40,
-5.6200e-40,  2.4820e-40,  3.1309e-40,
-3.1822e-17, -1.6585e-17, -8.8616e-18,
-5.9907e-17, -2.9812e-17, -1.6126e-17,
-2.4410e-16, -1.2541e-16, -6.7867e-17,
 1.5795e-01, -1.4429e-01, -6.0501e-02,
 5.9113e-02,  3.4391e-01,  1.4165e-01,
 5.2564e-02, -1.8209e-01, -6.8176e-02,
-7.7363e-41,  5.9969e-40,  5.9290e-40,
-7.4888e-41, -7.0945e-41,  5.3120e-40,
 1.3612e-40, -4.6718e-40, -1.0677e-40,
-1.1498e-01, -1.2925e-02,  2.6735e-02,
-8.1469e-02,  2.9678e-01,  1.8971e-01,
 2.0149e-02,  2.4207e-03, -1.2549e-01,
-6.6799e-02, -3.5900e-02, -5.6111e-02,
 9.5181e-02,  2.1216e-02,  2.0477e-01,
 8.5923e-03,  6.8615e-03,  3.8252e-02,
 4.5098e-03,  2.1321e-01,  3.4612e-03,
 3.5662e-01,  4.7532e-02,  2.5319e-01,
 4.1275e-02,  1.7951e-01,  3.2239e-02,
-2.6628e-21, -7.7165e-22, -4.9086e-22,
-1.4320e-21, -2.7134e-22, -1.2712e-22,
-1.9648e-21, -3.4172e-22, -1.3895e-22,
-2.2836e-40,  3.2091e-40, -4.4396e-40,
 2.9048e-40,  6.0866e-40,  3.7804e-40,
-3.0676e-40, -2.4897e-40,  4.9891e-40,
-1.8955e-28, -3.4994e-29, -1.2914e-29,
-4.7737e-29, -3.5212e-30, -6.4003e-31,
-8.2908e-29, -3.1692e-30, -3.6909e-31,
-9.3327e-02,  1.5314e-01,  1.0676e-01,
 2.5979e-01, -6.6826e-01,  2.3727e-01,
 1.4855e-01,  1.9205e-01,  8.8246e-02,
-5.5197e-40,  5.3162e-41, -5.2933e-40,
 1.0846e-41, -5.8128e-40, -3.1273e-40,
-2.8408e-40,  1.6989e-40,  4.8221e-41,
 7.8403e-02,  1.6407e-01,  7.9932e-02,
 3.2253e-01, -2.6036e-01, -8.9727e-02,
-7.5145e-02,  1.5536e-02, -8.2710e-02,
-2.1608e-01, -4.4619e-01, -4.4470e-02,
-3.9430e-01, -8.2373e-01, -7.0646e-01,
-6.9004e-03, -4.9697e-01, -1.4212e-01,
-1.8932e-06, -1.8356e-06, -1.6373e-06,
-1.9427e-06, -1.9113e-06, -1.7028e-06,
-1.8843e-06, -1.8616e-06, -1.6818e-06,
-4.7452e-29, -4.4894e-29, -2.5364e-29,
-5.6268e-29, -5.4363e-29, -3.0876e-29,
-4.3808e-29, -4.2767e-29, -2.4573e-29,
 3.8855e-40,  3.5152e-40, -4.8707e-40,
 4.3606e-41, -1.7886e-40,  5.1970e-40,
 6.2864e-40,  5.9972e-40,  2.2197e-40,
-2.1903e-37, -1.9174e-37, -7.0785e-38,
-2.7149e-37, -2.4810e-37, -9.5619e-38,
-1.8463e-37, -1.7136e-37, -6.7163e-38,
-2.9062e-30, -3.1324e-30, -1.0876e-30,
-2.7434e-30, -3.7036e-30, -1.2821e-30,
-6.8828e-31, -9.8708e-31, -3.7930e-31,
-6.3329e-41, -3.8604e-41, -2.8272e-40,
-3.3350e-40, -1.5210e-40, -4.2620e-41,
-1.7669e-41,  5.2291e-40, -3.3205e-40,
-3.0738e-25, -8.2305e-24, -2.1451e-23,
-1.4470e-24, -4.5131e-23, -1.2177e-22,
-4.2841e-24, -1.3077e-22, -3.5946e-22,
-8.5637e-08, -8.4715e-08, -7.7597e-08,
-8.7326e-08, -8.7480e-08, -8.0290e-08,
-8.4525e-08, -8.4963e-08, -7.8582e-08,
-5.8581e-27, -8.8483e-27, -8.1150e-27,
-7.4336e-27, -1.2036e-26, -1.1909e-26,
-6.6006e-27, -1.0685e-26, -1.0809e-26,
-5.6355e-40, -2.3469e-40, -3.5885e-40,
-2.0755e-40,  2.0377e-40,  3.2259e-40,
-5.3947e-40,  4.2747e-41,  4.8967e-41,
 4.5073e-41,  5.0069e-40,  2.6114e-40,
-4.8225e-40, -4.8317e-40, -5.4316e-40,
-5.4335e-40, -5.2994e-40,  2.6295e-40,
-1.1702e-40, -2.3137e-41, -4.5405e-40,
-4.6797e-40,  6.5582e-41,  1.8111e-40,
 6.1477e-40, -1.6827e-40, -2.0288e-40,
-2.4220e-41,  4.7774e-40,  5.1050e-40,
 4.9844e-40,  5.6437e-41,  4.7749e-40,
-6.8037e-41, -5.5944e-41, -5.2248e-40,
-2.9382e-40,  2.3800e-41,  1.5850e-40,
-4.5290e-40, -5.2260e-41,  2.3726e-40,
-1.9232e-40, -2.3502e-40, -2.9736e-40,
-2.8081e-40, -5.2929e-40, -4.0786e-40,
-3.0303e-41,  3.1336e-40, -5.8450e-40,
-1.5091e-40, -2.7371e-40, -4.5927e-40,
-4.0985e-38, -6.9102e-38, -5.4450e-38,
-6.2744e-38, -1.1526e-37, -9.9374e-38,
-4.8587e-38, -9.1819e-38, -8.0593e-38,
-2.9266e-29, -4.5005e-29, -3.9891e-29,
-3.8505e-29, -6.3370e-29, -6.0017e-29,
-3.2761e-29, -5.4145e-29, -5.1812e-29,
 3.3692e-40,  1.0044e-40, -6.6821e-41,
 9.2910e-41,  6.2137e-40, -3.5625e-40,
 1.8601e-40,  3.1653e-40, -1.1506e-40,
 1.2093e-40, -5.7191e-40,  5.6828e-40,
-2.3177e-40, -2.1648e-40,  5.3642e-40,
 4.8826e-40,  5.2760e-40, -4.9059e-40,
-2.0721e-40,  2.0122e-40, -5.9485e-40,
 3.8843e-40, -6.0861e-41, -4.0542e-40,
-3.4308e-40, -4.2822e-40, -3.9605e-40,
-5.7429e-40,  4.9242e-40, -5.9141e-40,
 4.6267e-40, -2.4953e-40, -2.9300e-40,
 5.3466e-40, -5.2403e-40,  3.5178e-40,
-1.8309e-40,  2.9157e-40, -7.7367e-41,
-5.8922e-40,  3.2359e-40, -6.1293e-40,
 6.1138e-40,  2.2121e-40, -5.0657e-42,
 4.7910e-40, -1.4080e-40,  1.9220e-40,
-3.5670e-40,  3.4204e-40, -5.0215e-40,
 1.1877e-41,  2.3114e-40, -4.7794e-40,
-3.6520e-40,  4.3222e-40, -5.2866e-40,
-6.0703e-40, -4.0896e-40, -1.2521e-40,
-4.1981e-40,  5.4404e-41,  3.3337e-40,
 1.3733e-01,  1.8485e-01,  7.6179e-02,
 8.1719e-02,  3.3343e-01,  2.9857e-02,
-4.2753e-03,  2.0957e-01,  1.8582e-02,
 2.9948e-07,  3.3403e-07,  3.7619e-07,
 3.4854e-07,  3.8224e-07,  4.1507e-07,
 3.7511e-07,  4.0398e-07,  4.3743e-07,
-1.7150e-41, -2.4088e-41, -1.5593e-40,
 6.3817e-41,  4.8004e-41, -1.1053e-40,
-2.5225e-40, -2.7111e-40, -4.2970e-40,
 1.0496e-06,  1.0916e-06,  1.1376e-06,
 1.1364e-06,  1.1756e-06,  1.2051e-06,
 1.1762e-06,  1.2105e-06,  1.2358e-06,
 1.0037e-02,  1.4957e-01, -4.9010e-02,
 2.6877e-02,  1.9067e-01, -1.9339e-03,
-2.2081e-02, -1.5137e-01, -1.6088e-01,
 1.6880e-41, -2.0352e-41, -4.1857e-42,
 2.0926e-40, -2.1394e-41, -5.4341e-40,
 4.6824e-40,  6.2682e-40,  4.9865e-40,
-3.2967e-01, -2.5981e-01, -1.3016e-01,
-2.6507e-01,  3.2282e-01,  4.3204e-01,
-7.0936e-02,  1.9800e-01,  9.4916e-02,
-1.0122e-02,  7.4127e-02, -7.1554e-02,
 7.7869e-02,  1.5734e-01,  1.3287e-01,
-9.5431e-02,  1.0984e-01, -7.6759e-02
}
,)"
R"(
{
-5.5262e-40,  3.7699e-40, -1.4920e-40,
 4.0064e-40, -2.0632e-40, -4.4801e-41,
-3.6749e-40,  5.9043e-40, -1.5942e-40,
-5.9219e-42, -4.1286e-40, -1.6920e-40,
-2.5927e-40, -4.5458e-41,  2.0990e-40,
-4.6860e-40,  5.0483e-40,  2.8004e-40,
-4.0641e-40,  6.0770e-40, -3.8297e-42,
 5.7537e-40,  5.7772e-40, -1.0048e-40,
 1.5945e-40,  3.9582e-40, -2.6190e-40,
-5.1046e-40, -5.5028e-40,  5.8786e-40,
-3.5033e-40, -1.2031e-40, -3.4156e-40,
 3.0058e-40,  4.3043e-40,  5.9825e-40,
 4.9197e-40,  2.5974e-40, -4.3461e-41,
-4.1935e-40, -1.6383e-41, -1.4680e-40,
-5.3501e-40, -2.6348e-40,  3.0631e-40,
-5.2019e-40, -4.4123e-40,  2.3984e-40,
-4.4682e-41, -4.6000e-40, -5.0418e-40,
-4.1263e-40,  4.5391e-40,  2.8844e-40,
 5.2179e-40, -1.3188e-40,  5.1600e-40,
-2.2913e-40, -3.1127e-40,  5.4478e-40,
 2.3395e-41,  5.4758e-40,  2.0998e-40,
-1.9914e-10, -2.0700e-10, -1.9815e-10,
-2.1098e-10, -2.1989e-10, -2.1131e-10,
-2.0797e-10, -2.1693e-10, -2.0860e-10,
-2.1061e-40, -2.1208e-40, -3.3698e-40,
 3.2370e-40,  2.9276e-40, -3.6860e-40,
 3.4752e-40, -2.0660e-40, -3.8183e-40,
-8.0136e-02,  1.3809e-02,  1.6846e-03,
 3.7960e-02,  8.7557e-02, -3.5498e-01,
 9.8165e-03,  9.8384e-02,  1.2395e-01,
-2.8751e-02,  9.9172e-02,  5.5841e-02,
-4.0383e-02,  1.0856e-01, -5.4339e-01,
 1.3245e-02, -4.7642e-02, -1.0427e-01,
-7.4696e-03,  5.0806e-02, -1.7179e-01,
 5.0303e-02, -4.0322e-01,  7.4760e-01,
-9.2342e-02,  1.1958e-01, -1.8871e-01,
 3.7044e-40, -4.6951e-40, -1.9873e-40,
 5.3289e-41,  2.7689e-40, -4.6994e-41,
-3.1404e-40, -5.9106e-40,  6.0436e-40,
-6.0294e-40, -3.6565e-40, -1.1884e-40,
 5.5933e-40, -9.5741e-41,  4.4736e-40,
 4.3267e-40, -4.9583e-40,  3.4437e-40,
-1.7432e-40,  1.4518e-40,  2.1033e-40,
-3.4667e-40,  1.7222e-40, -2.5651e-40,
-5.2517e-40,  2.8983e-41, -1.3832e-40,
-1.4153e-01,  9.4023e-02, -9.8526e-02,
 2.0678e-01,  4.0842e-01, -1.1853e-01,
-1.4108e-01, -1.1005e-01, -8.1274e-02,
 3.4336e-41,  1.5625e-40,  2.7213e-40,
-5.3447e-40, -3.7330e-40, -3.3637e-40,
-4.3563e-40, -3.7094e-40,  1.2820e-41,
-8.1700e-02, -1.8215e-01, -1.6011e-01,
-1.4203e-01,  5.3791e-02, -3.7663e-02,
-1.1705e-01, -1.2604e-01, -8.4890e-03,
-6.1578e-02, -3.3907e-01,  2.2344e-03,
 1.5060e-01, -1.9199e-01, -5.5274e-02,
 6.2300e-02,  9.1084e-02,  1.3788e-02,
 4.9025e-02,  3.3738e-01, -1.8104e-01,
-2.5051e-01,  8.2363e-02,  2.0325e-01,
 5.6988e-02, -1.5118e-01,  6.8897e-02,
-4.6233e-40,  1.2244e-40, -3.9802e-40,
 5.8530e-40, -2.4162e-40,  4.6793e-40,
-4.8362e-40,  3.3071e-40,  1.7094e-40,
 3.5249e-40, -4.8579e-40,  1.9374e-40,
 6.2372e-42,  5.8402e-41,  3.2851e-40,
 6.1488e-40,  1.8086e-40, -5.2451e-40,
-3.0723e-40, -5.6704e-40, -5.9899e-40,
-3.5975e-40, -1.3818e-40, -2.7285e-40,
 2.4468e-40,  8.3606e-41,  1.8818e-40,
-2.3749e-01, -2.7008e-01, -1.5222e-03,
 1.4806e-01,  9.0783e-02,  2.7170e-02,
 1.8706e-01,  1.8162e-01, -1.1799e-01,
-1.9852e-40, -4.8879e-40, -3.1971e-40,
-1.0245e-40,  9.1421e-41,  5.3018e-40,
 2.2240e-40, -1.4666e-40, -4.4259e-40,
 1.1835e-01, -2.7624e-01,  1.1446e-01,
 1.3574e-01,  4.3109e-01,  1.3227e-01,
 3.2554e-02,  1.7139e-01, -1.1988e-01,
 3.5376e-02,  8.9191e-02,  6.7643e-02,
-8.2716e-02,  2.4178e-01,  6.0818e-02,
-6.7722e-02, -3.3712e-02,  3.0664e-02,
-6.6948e-02,  2.2886e-01,  1.8143e-01,
 1.8636e-01, -2.4800e-01,  1.7185e-01,
-6.5479e-03,  1.8828e-01, -7.4464e-02,
-2.8281e-30, -5.8969e-31, -2.3180e-31,
-1.6163e-30, -3.8426e-31, -1.6788e-31,
-1.9412e-30, -4.1995e-31, -1.7651e-31,
-2.0525e-40,  4.6680e-40,  5.9108e-41,
 1.0336e-40, -5.7226e-41, -6.1906e-40,
-1.8693e-40,  5.5777e-40,  6.0898e-40,
-3.4735e-41, -3.2674e-40, -2.3864e-41,
-3.3596e-40,  3.3107e-40,  1.0843e-40,
 5.1103e-40,  6.0598e-40, -3.6267e-40,
-4.5583e-03, -1.0635e-01, -7.4962e-02,
-1.2741e-01,  2.7234e-01,  1.0508e-01,
-2.1207e-01,  9.6720e-02,  3.4641e-02,
 1.1304e-12,  1.1614e-12,  9.7086e-13,
 1.3361e-12,  1.3697e-12,  1.1286e-12,
 1.2620e-12,  1.2938e-12,  1.0680e-12,
-8.4197e-02,  6.3834e-02,  2.3157e-02,
-2.1280e-02,  2.9074e-01,  8.5883e-02,
-1.3695e-01, -1.6047e-01, -4.5834e-02,
-1.3848e-01, -6.6090e-02, -7.7201e-02,
-5.1963e-02,  6.0643e-02, -4.9932e-02,
 1.1779e-01,  1.7521e-01,  3.0366e-02,
 4.7601e-03,  4.3941e-02, -3.5985e-02,
 1.7692e-02, -2.3705e-01,  2.1062e-01,
 7.7174e-02, -7.6616e-02,  2.0102e-02,
-3.6353e-06, -3.5534e-06, -3.2461e-06,
-3.6813e-06, -3.6196e-06, -3.3222e-06,
-3.5581e-06, -3.5179e-06, -3.2504e-06,
-7.3892e-11, -7.2930e-11, -6.8104e-11,
-7.9244e-11, -7.7770e-11, -7.2319e-11,
-7.7297e-11, -7.5673e-11, -7.0195e-11,
-1.5180e-10, -1.5027e-10, -1.4244e-10,
-1.6013e-10, -1.5761e-10, -1.4940e-10,
-1.5682e-10, -1.5395e-10, -1.4553e-10,
-9.1167e-02,  1.2374e-01, -3.8304e-02,
 2.2641e-01,  2.4855e-01, -4.3174e-02,
 1.4364e-01,  1.8438e-01,  1.1617e-02,
 6.1925e-40,  3.3333e-40,  1.8962e-40,
 3.2481e-40, -1.7566e-40, -3.0456e-40,
 2.7654e-40,  3.8422e-41,  4.9191e-40,
 7.5657e-02, -1.0697e-03,  3.0319e-02,
-4.7642e-02, -9.4454e-02, -2.6543e-02,
-5.3129e-02, -1.9667e-01, -1.0851e-01,
-8.5909e-03,  1.2177e-01,  2.6434e-01,
 2.4468e-02,  5.0484e-02,  3.4698e-01,
-1.4764e-03,  3.7374e-02,  1.2658e-01,
 2.0602e-02, -2.4624e-02,  1.3741e-01,
 1.8641e-02,  4.0484e-01,  3.2976e-01,
-4.4809e-01, -3.2104e-03,  1.6290e-03,
 8.1306e-41,  2.0311e-40,  2.9683e-40,
-5.7636e-40,  4.4291e-40,  4.3356e-40,
-7.1797e-41,  4.5366e-40,  3.9953e-40,
-4.5418e-40,  4.1805e-40, -3.2458e-41,
-9.4881e-41, -8.6365e-41, -1.9294e-40,
 7.1954e-41, -9.8565e-41, -5.5540e-40,
-5.3769e-40,  1.4094e-40, -1.5355e-40,
 8.8038e-41, -3.6848e-40, -1.2237e-40,
-2.8267e-41, -1.7583e-40, -5.9647e-40,
 1.0929e-01,  2.9895e-02, -1.4923e-01,
-1.1234e-01, -1.0514e-01, -1.3280e-02,
 2.2255e-01,  6.4152e-03, -1.6309e-02,
-1.5899e-40, -7.2549e-41, -2.6734e-40,
-3.3842e-40,  3.3255e-40,  4.2694e-40,
 5.2940e-40,  3.2455e-40, -3.7081e-40,
 6.3639e-02, -3.3720e-02, -2.3453e-02,
 1.9477e-01,  5.2267e-02,  1.8565e-02,
 1.6048e-01,  2.7636e-01,  1.5930e-02,
 1.7673e-03,  6.3646e-02, -1.5127e-02,
-3.7787e-02, -1.4037e-01, -3.6231e-02,
-1.5636e-02, -7.8742e-02, -2.4137e-02,
-5.0748e-02,  6.5641e-02, -2.5353e-03,
 8.4955e-02,  7.4231e-01,  1.3795e-01,
-1.4552e-01,  2.0869e-01,  4.0739e-02,
-2.0015e-41,  5.2988e-40,  2.7578e-40,
 4.1051e-40,  1.2834e-40, -3.4898e-40,
-1.1975e-40,  4.2374e-40, -3.0404e-41,
-6.3014e-40,  4.6330e-40, -4.4141e-41,
 2.5442e-41,  5.7456e-40,  2.3848e-40,
-1.0788e-40, -5.0563e-40, -5.3638e-41,
 3.5728e-40,  1.9752e-40,  6.1004e-40,
 2.8189e-41, -6.2151e-40,  1.1807e-41,
 6.5305e-41,  5.2028e-40,  1.3692e-40,
 6.4391e-02, -1.3079e-01, -3.7980e-02,
-3.2362e-01, -3.7239e-01, -8.0182e-02,
-2.6787e-01, -3.1240e-01, -1.2798e-02,
-1.2072e-40,  5.3996e-40, -3.4352e-40,
-8.0996e-41, -3.0208e-40,  3.1848e-40,
-5.6407e-40,  2.4674e-41, -2.1055e-40,
-9.2897e-02,  1.8040e-01, -4.3269e-01,
-7.6669e-02,  4.3554e-01, -4.4870e-02,
-2.3249e-02, -1.1805e-01,  1.0507e-01,
-5.2540e-02, -3.6856e-01,  1.1246e-01,
-2.3632e-02,  1.3165e-01, -1.5380e-02,
-1.1467e-02, -5.3754e-02, -4.1619e-02,
-1.5635e-01,  3.8584e-01, -1.4434e-01,
 1.7523e-01,  3.7253e-02,  4.9784e-01,
 5.8484e-02, -8.4711e-02, -7.7498e-02,
-1.6956e-40,  5.4293e-41, -2.5140e-40,
-3.1995e-40, -4.8337e-40,  2.5539e-40,
-1.1449e-40,  1.9503e-40, -1.7368e-40,
 5.4753e-40,  5.9720e-40, -4.7821e-40,
 3.8830e-40, -3.1984e-40, -2.7163e-40,
-5.3411e-40,  7.2638e-41,  4.3186e-40,
 4.6654e-40, -5.9540e-40, -2.8155e-40,
-1.4801e-40, -1.6945e-40,  1.9723e-40,
 5.8380e-40, -6.1587e-40,  3.3667e-40,
-2.9327e-02, -4.2746e-02, -1.5018e-01,
 8.6354e-02,  2.8140e-01,  1.2970e-02,
-2.0755e-01,  6.7548e-02, -3.6049e-02
}
,)"
R"(
{
 9.5728e-41,  5.3991e-40, -1.3764e-40,
-2.0389e-40,  2.4254e-40,  3.3492e-40,
 6.5289e-41, -3.0842e-40,  5.5850e-40,
 7.7599e-02,  2.5043e-02, -1.4099e-02,
-3.3184e-02,  5.6863e-01, -2.7001e-02,
-5.2659e-02,  5.4713e-02,  2.3991e-03,
 2.2010e-02, -3.9120e-02, -1.1558e-01,
 9.1633e-02,  1.3070e-01,  1.2489e-01,
-4.4040e-02, -1.6324e-02, -4.9631e-02,
-7.3548e-02, -2.0492e-01,  1.4043e-01,
-6.0411e-02,  5.7710e-02, -3.6840e-02,
 1.3173e-02,  2.3215e-03,  1.1820e-02,
 2.5772e-02, -1.3436e-01, -5.9285e-02,
-9.3983e-02,  1.1545e-01,  1.1602e-01,
-1.8505e-02,  6.1498e-02, -1.3097e-02,
 9.8690e-03, -2.1338e-02, -1.2175e-01,
 1.7936e-02, -2.7811e-02,  6.7037e-02,
-5.1401e-03,  7.6421e-02, -1.0794e-01,
 4.6409e-02,  3.4701e-01,  2.6587e-02,
 8.4175e-02,  5.2712e-01,  6.8999e-02,
-8.0756e-02,  1.9648e-01, -8.4639e-02,
 1.2818e-01,  4.0660e-02,  7.6715e-02,
 8.7991e-02,  4.6556e-01, -4.0025e-02,
 2.1251e-03, -8.3784e-03,  5.9859e-02,
 1.9835e-40, -3.4675e-40, -7.9692e-41,
-1.4304e-40,  2.3927e-40, -5.9796e-40,
 3.8209e-40, -6.3260e-41, -9.2501e-41,
 3.2007e-01,  1.5800e-01, -1.9594e-02,
-4.5315e-02,  1.0536e-01, -8.0692e-02,
 2.1185e-01, -3.1418e-01, -1.5257e-01,
 8.6294e-02, -1.3398e-01, -1.0694e-01,
 8.6084e-02, -1.2393e-03,  1.7549e-02,
-1.5504e-01, -1.3112e-01, -3.5905e-02,
-3.8190e-01,  3.8393e-01,  1.6587e-02,
 1.5002e-01,  1.9586e-01, -2.6260e-01,
-4.0159e-02, -8.2891e-02, -1.7761e-01,
-1.8611e-01, -1.1241e-02, -4.2538e-02,
-5.7898e-02,  2.4583e-01,  4.1590e-02,
 2.4890e-02,  7.9409e-03, -2.7418e-02,
 6.6194e-03, -4.2441e-02, -1.1167e-01,
-1.3236e-01, -7.9642e-02, -6.0623e-02,
-4.7198e-03,  5.6904e-02,  1.2651e-01,
 1.2925e-01, -5.9162e-02, -9.1949e-04,
 1.8668e-02, -2.6361e-02, -7.1042e-03,
-4.3178e-02,  2.6050e-04,  4.4799e-02,
 7.9674e-02,  2.7656e-02,  7.1211e-03,
 1.1463e-01,  1.0765e-01,  7.6066e-02,
-8.0780e-02, -5.4875e-02,  1.5209e-02,
-3.7365e-13, -3.7819e-13, -3.5929e-13,
-4.0298e-13, -4.0881e-13, -3.9033e-13,
-3.9409e-13, -3.9950e-13, -3.8277e-13,
-1.7847e-02, -1.7537e-02, -3.7313e-03,
 2.6531e-02,  7.5951e-02, -4.0134e-03,
 1.7387e-02,  6.0044e-02, -9.0211e-02,
 2.7091e-02,  8.8333e-02,  1.0619e-01,
 5.0470e-02,  1.2406e-02,  1.5503e-01,
-1.5936e-02, -2.2422e-01, -2.4640e-02,
-8.2430e-03, -1.4097e-02, -6.2474e-02,
 8.0534e-02,  1.8603e-01, -3.1725e-02,
-3.1621e-03,  2.0362e-03, -1.4002e-01,
-7.3799e-03,  1.5881e-01,  6.7195e-02,
 4.5946e-02,  2.4358e-01,  1.4677e-01,
-7.4788e-02,  6.7297e-02,  9.0735e-02,
-8.4553e-03, -1.1877e-02,  4.4209e-02,
-1.4281e-02, -6.8849e-02, -4.1386e-03,
 3.2286e-02,  4.7128e-02, -1.2988e-02,
-2.2990e-02, -8.9265e-02,  6.4050e-02,
-2.3354e-02,  1.3846e-01, -1.6256e-01,
-6.5661e-02, -2.8983e-02, -4.3497e-02,
 1.0597e-02, -2.3534e-02, -2.6068e-02,
-7.8812e-02,  1.9502e-01,  6.8938e-03,
 3.2025e-02,  2.3353e-02,  4.9225e-02,
-5.0273e-40,  1.2403e-41,  5.8127e-40,
 3.2777e-40, -3.5740e-40,  4.9781e-40,
-2.4198e-40, -4.6311e-40,  1.3330e-40,
-3.0803e-01,  1.7804e-01,  1.0604e-01,
 4.1405e-01,  1.9740e-01, -5.3067e-02,
 2.3738e-01, -1.6828e-01,  1.5338e-01,
 6.6857e-03,  1.8623e-01, -1.2126e-01,
-1.6323e-01, -1.2719e-02, -1.7743e-01,
-1.3612e-01, -3.4442e-02, -1.0552e-01,
-1.4560e-01,  1.8771e-01,  8.4508e-02,
 5.8732e-02, -2.2378e-01,  1.2673e-01,
 3.0455e-03,  3.8438e-02, -6.2235e-02,
 1.9951e-02,  2.6963e-01, -1.8594e-01,
-8.6550e-02, -1.3097e-01, -3.5032e-02,
 2.0423e-02, -9.0499e-02,  1.7130e-01,
-1.8592e-01,  6.6808e-02, -1.5768e-01,
-6.4402e-02, -1.2265e-01,  6.8487e-02,
 1.9899e-02,  9.3376e-02,  7.8577e-02,
-1.3384e-01, -7.6429e-02,  1.7142e-02,
-1.2385e-01, -1.1821e-01, -1.2716e-03,
 5.3770e-02,  1.4973e-01,  1.4762e-01,
-4.7688e-02, -1.1733e-01, -1.5032e-01,
-2.0699e-01, -9.4949e-02, -2.6374e-02,
 4.4489e-02,  1.8376e-02, -7.6844e-02,
 1.8831e-40, -2.6056e-40, -4.7602e-40,
-3.4079e-40,  1.5054e-40,  1.2387e-40,
 2.3040e-40,  1.4644e-40,  5.6365e-40,
-2.0809e-02,  5.3674e-03,  1.7057e-03,
 2.4160e-01,  4.1348e-01,  3.5215e-02,
 8.2154e-02,  2.0431e-01,  1.0366e-01,
-1.5149e-02,  1.0521e-01, -4.1706e-02,
-5.0651e-02,  2.3615e-02, -9.3860e-02,
-1.0823e-01, -6.3645e-02, -1.1573e-01,
-2.4116e-02,  1.3546e-02, -1.0298e-03,
 1.2102e-02,  2.2630e-02,  1.1375e-01,
 1.3966e-02,  1.0754e-01,  1.6621e-01,
 1.6213e-02,  2.0816e-01,  8.9441e-02,
-7.5452e-02,  3.4580e-03, -3.3317e-01,
 5.0917e-02,  1.3898e-01, -1.0723e-01,
 6.0473e-03,  8.9741e-02, -6.8206e-02,
-7.1770e-02, -3.5661e-01, -2.8935e-01,
-1.6324e-02,  2.5728e-02, -1.1281e-02,
-1.3390e-01, -9.3090e-02,  4.3366e-02,
 4.8620e-02,  1.4917e-01,  1.6295e-01,
 2.4123e-03, -7.6347e-02, -8.0226e-02,
 6.0740e-03,  3.7065e-02,  4.5518e-04,
-1.3793e-01,  2.3848e-01, -1.1199e-01,
 1.0422e-01,  1.1214e-01,  3.3457e-02,
-3.2827e-40,  5.9135e-40,  3.3773e-40,
-5.8903e-40, -5.9439e-41,  1.9973e-40,
-3.6141e-40, -4.7563e-40, -1.0222e-40,
 7.3457e-02, -8.2031e-02, -2.9504e-02,
-5.3420e-02,  4.9697e-02,  7.6779e-03,
 2.1180e-02,  1.1069e-02, -1.1940e-02,
 1.7302e-02,  9.9063e-02,  4.8847e-02,
 4.9513e-02,  2.4240e-01,  2.7174e-01,
 2.7487e-01,  1.9410e-01,  3.1165e-01,
-6.7532e-03, -1.1608e-01, -5.0876e-02,
 1.2107e-01,  3.1073e-01,  7.1681e-02,
-1.1411e-01, -1.7902e-01,  7.8898e-02,
-2.0117e-02,  3.6394e-01,  1.4546e-01,
-8.0861e-03, -4.3956e-02, -1.3473e-01,
 5.1519e-02, -3.1122e-01, -4.6847e-02,
 5.0405e-02, -1.0611e-02, -1.0557e-01,
-4.4346e-02, -1.4505e-01,  5.3977e-02,
-2.6288e-01,  1.8247e-02, -1.1606e-01,
 1.0706e-01, -9.3675e-02,  1.1757e-01,
-5.0440e-02, -1.1784e-01, -4.0599e-02,
 1.9618e-01,  9.9370e-02,  8.2258e-02,
 2.6762e-02, -5.0740e-02, -1.8302e-02,
 5.3340e-02,  6.5710e-02,  6.1552e-03,
-7.2158e-02, -3.5563e-02,  8.2140e-02,
 3.1534e-40,  3.6427e-40,  3.0437e-40,
 4.2856e-41, -4.7870e-40,  5.6317e-40,
-2.4673e-40, -6.9736e-41,  8.1050e-41,
 1.4544e-01,  8.2490e-02, -9.2349e-03,
 2.6124e-01,  2.7494e-01, -5.4946e-02,
 1.8233e-01,  1.2428e-01, -6.7498e-03,
 9.7639e-02, -6.2085e-03,  4.8154e-02,
 2.7379e-02, -1.8443e-01,  4.0402e-02,
 1.8893e-03, -5.2282e-03,  6.7548e-03,
-1.6559e-01,  9.7901e-02, -1.1869e-01,
-2.1287e-01,  4.1023e-01, -9.7379e-02,
-1.3767e-03, -1.6343e-01, -9.5059e-02,
-1.3547e-01,  2.0094e-01,  1.0102e-01,
-2.1311e-01, -1.5088e-01,  1.8175e-01,
 4.6946e-02, -1.3963e-01,  1.0220e-01,
 1.7536e-01, -2.4758e-01, -1.1481e-02,
 6.1596e-02, -4.0352e-01, -1.4348e-01,
 3.1690e-02,  1.7240e-01,  7.0780e-02,
 9.9953e-02, -1.4154e-01, -8.3038e-02,
 1.4527e-01, -2.1430e-01, -7.5840e-02,
 1.6146e-01,  3.7508e-02,  5.3833e-02,
 1.6723e-01,  1.7113e-01, -4.8512e-02,
 2.1319e-01,  4.7031e-01,  1.1570e-01,
 2.0330e-01,  2.4636e-01,  6.9924e-02,
-2.1165e-40, -1.9259e-40, -5.0990e-41,
-7.1298e-42, -4.2590e-41,  3.1709e-40,
 4.1065e-40, -4.2585e-41,  3.4243e-40,
-1.0338e-40,  4.6039e-40, -3.3818e-40,
-3.9589e-41,  5.9574e-40, -5.8014e-41,
 1.4505e-41, -3.5326e-40, -3.9806e-40,
 4.2423e-40, -1.7055e-40, -4.9666e-40,
 2.2853e-40, -2.4684e-40, -1.3794e-40,
-5.6764e-40, -1.7905e-40, -5.8915e-40,
-1.4755e-27, -2.0405e-28, -4.8677e-30,
-7.1151e-28, -9.7603e-29, -3.5264e-30,
-2.7455e-29, -5.7734e-30, -2.8633e-31,
-5.9960e-06, -5.9595e-06, -5.8686e-06,
-6.0381e-06, -6.0191e-06, -5.9605e-06,
-5.9849e-06, -5.9981e-06, -5.9654e-06,
-4.8277e-22, -7.0529e-22, -8.7179e-22,
-4.6334e-22, -6.3505e-22, -8.8438e-22,
-3.3883e-22, -4.2421e-22, -5.9002e-22,
-2.9574e-40,  4.0860e-40, -1.5966e-40,
-6.7527e-41,  7.6661e-41, -5.9491e-40,
 3.0843e-40,  8.1079e-41, -2.5140e-40,
-3.7315e-40,  9.4787e-41,  4.6794e-40,
 1.9383e-40,  5.0336e-41,  3.0561e-40,
-5.4286e-40,  5.5999e-40, -4.6977e-40
}
,)"
R"(
{
-1.7778e-01,  5.2351e-03,  1.6035e-02,
-9.7482e-02, -1.1056e-02, -5.0999e-02,
 1.7460e-01, -4.0005e-02, -5.0911e-02,
-9.3843e-02,  1.2640e-01, -1.5016e-02,
-5.2880e-01,  1.9469e-01, -9.0037e-02,
-8.9136e-02,  9.8632e-02, -1.5009e-01,
-1.8080e-01,  1.1396e-01, -2.6178e-02,
-1.6689e-02,  1.4132e-01, -6.7769e-03,
-2.1120e-02,  6.8616e-02, -7.8209e-02,
 4.8237e-02, -2.5303e-02,  1.7882e-02,
-4.2852e-02, -1.5071e-02, -3.3818e-02,
 1.3635e-01,  4.5330e-01,  2.1489e-01,
 2.7362e-02, -7.4152e-02,  2.3185e-03,
 1.8771e-01, -2.0827e-02, -7.5581e-02,
 1.4675e-01, -6.5552e-02,  4.2292e-02,
 1.3990e-01, -4.1598e-01,  2.1609e-03,
 1.5997e-01,  1.1375e-01, -1.8272e-02,
 1.9045e-02, -4.2702e-02, -2.5602e-02,
 1.6432e-01, -1.2783e-01, -1.8285e-03,
 2.9414e-01,  1.7401e-01, -2.6321e-01,
-1.0125e-01,  1.3565e-01,  1.5894e-02,
-3.7351e-40,  6.3010e-40, -1.2071e-40,
-4.6380e-40,  1.8442e-40, -3.5994e-40,
-2.1459e-40, -4.3455e-40, -6.1978e-41,
-2.3638e-40, -4.6965e-40, -3.4232e-40,
-1.6517e-40,  4.7178e-40, -1.6757e-40,
 6.7890e-41, -4.3000e-40,  1.8323e-40,
 4.5416e-40, -2.9010e-40, -1.5200e-40,
-3.5533e-40, -8.7351e-41,  6.5595e-42,
 5.1625e-40, -6.0418e-40, -2.7846e-40,
-2.1861e-10, -2.2422e-10, -2.1298e-10,
-2.2653e-10, -2.3500e-10, -2.2512e-10,
-2.1802e-10, -2.2681e-10, -2.1608e-10,
-3.2862e-40,  3.4241e-40, -1.3264e-40,
 2.8762e-40,  1.3843e-40,  3.0949e-40,
-3.7702e-40,  2.6194e-40,  2.1451e-40,
-3.2283e-40, -5.5487e-40,  5.8744e-40,
 1.6124e-40,  3.3512e-40,  3.1454e-40,
-3.5417e-40, -5.7692e-40,  5.5184e-40,
 3.5641e-40, -4.3187e-40, -3.5314e-40,
 4.9246e-40,  5.9593e-40,  8.3132e-41,
-2.3841e-40, -5.6196e-40, -3.2230e-41,
 4.3824e-40, -3.8344e-40, -9.9086e-42,
-2.9323e-40,  2.1916e-40,  4.4739e-40,
 5.6837e-41,  5.1796e-41, -2.4338e-40,
-2.2853e-40, -3.8920e-40,  6.1587e-40,
-2.9474e-41,  4.6214e-40, -3.6292e-40,
-1.4928e-40, -3.6708e-41,  5.2020e-40,
-1.2983e-12, -2.6539e-12, -1.9817e-12,
-6.5613e-12, -1.0255e-11, -6.6919e-12,
-8.3217e-12, -1.7832e-11, -1.1086e-11,
-4.9138e-40, -9.0061e-42,  4.6251e-40,
-2.9970e-41, -2.5468e-40, -3.5660e-40,
 2.5450e-40, -9.5634e-38, -3.2369e-32,
-1.0233e-06, -8.2108e-07, -1.1668e-06,
-5.9592e-07, -3.9529e-07, -5.7435e-07,
-6.0253e-07, -3.8785e-07, -4.9365e-07,
-8.9372e-37, -2.1590e-36, -2.1060e-40,
-1.5666e-35, -1.1466e-38, -2.3366e-40,
-5.4077e-38,  5.0487e-40, -3.3736e-40,
-1.5357e-13, -8.4607e-14, -1.9206e-16,
-5.5373e-13, -3.0787e-13, -1.0513e-15,
-1.0468e-13, -8.6069e-14, -2.2453e-16,
-4.7501e-14, -1.3426e-13, -1.1133e-13,
-1.3801e-14, -2.4024e-14, -3.5120e-14,
-1.9817e-17, -1.3229e-17, -3.2854e-17,
-1.4365e-18, -4.1143e-15, -9.2614e-14,
-1.1174e-19, -1.6235e-15, -1.5600e-13,
-1.2643e-21, -3.9578e-17, -1.2038e-14,
-2.9789e-40, -4.6452e-40,  1.5649e-40,
-1.8445e-40, -5.2942e-40,  2.5130e-40,
 6.2269e-40,  3.9166e-41, -2.4197e-40,
 9.0835e-02, -5.2035e-03, -2.5980e-02,
-1.0090e-01, -7.4167e-02,  1.3364e-01,
 1.0302e-01, -1.5250e-01,  1.2417e-01,
 4.7205e-02, -2.3839e-01, -1.4983e-02,
 5.6824e-02, -1.8259e-02,  9.6426e-02,
 5.9740e-03, -1.4198e-01, -2.1076e-01,
-1.5837e-01,  6.4749e-02, -2.1417e-01,
-3.4048e-02,  4.9638e-01,  2.0984e-03,
-1.4335e-01,  4.8295e-02, -9.2209e-02,
 1.9450e-01, -1.3603e-01,  1.2008e-01,
 1.6803e-01,  5.6805e-02,  1.1518e-01,
 5.9320e-02, -3.8200e-02, -1.1340e-01,
-8.6877e-02,  1.1533e-01, -4.9870e-02,
-7.2811e-03,  2.5730e-01, -1.8536e-01,
-6.4965e-02,  1.0364e-01,  1.3706e-02,
 4.6974e-02, -1.0049e-01, -1.7460e-01,
-1.7910e-01,  3.0771e-01, -2.5757e-01,
-2.2846e-02, -3.7491e-03, -5.2171e-03,
-4.7762e-02, -4.7776e-02,  5.1125e-01,
-2.0210e-01,  6.4815e-02, -6.1606e-02,
 7.3686e-04, -1.6226e-01, -3.0327e-02,
 5.6501e-40,  5.2828e-40, -5.9773e-40,
-4.3530e-40, -1.1658e-40,  4.9705e-41,
 4.8101e-40,  5.0236e-40,  2.0476e-40,
-1.1412e-01,  1.3391e-01, -1.2279e-01,
 1.4370e-01,  3.7617e-01,  7.1407e-02,
 6.9661e-02,  3.1963e-01, -1.7089e-02,
-4.7530e-02,  6.5411e-02, -2.4915e-02,
 3.3429e-02, -1.3899e-01, -3.3875e-02,
-1.9261e-02, -1.3162e-01,  1.1415e-01,
 2.0599e-02, -3.8667e-02, -7.2190e-02,
-2.1112e-01, -1.6525e-01, -2.3430e-02,
-1.2287e-02, -2.6637e-01,  1.0859e-03,
-2.8564e-02,  4.8846e-02,  4.2412e-02,
 1.4632e-01,  1.5974e-02, -1.0699e-01,
 5.5661e-02, -2.0952e-01,  2.4151e-02,
-2.3510e-02, -5.0570e-02,  1.0799e-01,
 1.7495e-01, -1.5788e-03, -1.6447e-02,
 7.7642e-02, -9.3888e-02,  1.3891e-03,
 2.2658e-02,  1.4058e-01,  1.0639e-01,
-5.5626e-02, -3.0794e-01, -5.7160e-02,
 1.0874e-01, -8.3907e-02,  4.2106e-02,
 1.7688e-02,  1.8090e-01, -2.1718e-03,
-1.0659e-02, -2.1302e-01,  1.0056e-01,
-6.0693e-02, -2.3624e-02,  6.3688e-03,
-2.7320e-40, -1.3336e-40,  2.4202e-41,
-7.1225e-41,  1.2848e-40,  1.5426e-40,
-4.2798e-40,  6.5075e-41,  6.2629e-40,
 1.6905e-01, -1.7379e-01, -2.1360e-02,
-2.9396e-01,  1.1782e-01,  7.9111e-02,
-6.4767e-03, -1.9949e-01,  5.4243e-02,
-3.2753e-02, -1.5810e-01,  5.2257e-02,
-1.8133e-02,  2.0548e-01, -2.8071e-01,
-5.3725e-02,  8.4067e-02, -7.4639e-02,
 8.9137e-02, -2.3078e-01, -1.9626e-01,
 3.1276e-01,  1.5332e-01, -1.9590e-01,
-1.8318e-02,  6.8460e-02,  9.1476e-03,
 8.2398e-02,  8.5883e-03,  7.6830e-02,
-1.4580e-01,  4.6253e-01, -3.1900e-01,
-1.1051e-01,  6.3807e-02, -2.5130e-02,
-1.2029e-01, -3.8982e-03,  2.1654e-02,
-3.2017e-01,  2.0265e-01, -1.7311e-01,
-1.3229e-02,  1.3805e-01, -6.2689e-02,
-3.6619e-02, -1.9366e-01,  2.7177e-01,
 5.5937e-02,  7.9713e-02, -2.3872e-01,
-3.9690e-02,  2.2914e-02, -1.7779e-02,
 1.1110e-01,  1.6618e-01,  3.6139e-01,
 7.9777e-02,  4.3655e-01,  3.0597e-01,
-5.5125e-02,  6.1229e-02,  1.2414e-01,
 2.1644e-40,  7.2343e-41,  5.5580e-40,
-4.3927e-40,  5.0561e-40, -1.5560e-41,
-3.2783e-40, -8.8219e-41,  5.4415e-40,
-6.7176e-02, -3.4930e-02, -2.7087e-02,
 1.0489e-01,  2.1178e-01, -1.6752e-01,
-1.2627e-01, -2.4207e-01, -7.4667e-02,
-3.1470e-02, -1.3365e-02,  8.7742e-02,
-2.2809e-02, -4.7991e-01,  2.4740e-02,
 6.4418e-02,  3.4818e-02, -2.9275e-01,
-2.8830e-01, -7.0458e-02,  7.8922e-02,
-1.4436e-01,  4.1068e-02,  6.2896e-02,
 4.1061e-03,  2.1844e-01,  9.0488e-02,
-1.1085e-01,  8.3761e-02,  3.2634e-02,
 3.2470e-01, -2.7760e-01,  4.1235e-02,
 8.6625e-02,  2.6816e-01, -1.3560e-01,
 3.8789e-01,  3.2406e-01,  1.0631e-01,
 7.5131e-02, -2.0206e-01,  1.3027e-01,
 4.0382e-02,  2.4350e-01, -3.6042e-03,
-1.0063e-01,  1.9418e-01, -7.7039e-02,
 9.4531e-03,  7.1605e-02,  1.4004e-01,
-2.0591e-02,  4.5944e-02, -2.6721e-03,
-3.4665e-03,  2.2560e-01, -8.2930e-02,
-1.5507e-01,  2.7206e-01, -2.8665e-02,
-3.4909e-03,  1.7696e-02, -8.5492e-02,
 2.1541e-40, -3.3029e-40,  1.7678e-40,
-3.9857e-40, -1.1965e-40, -8.6754e-41,
-4.0721e-40,  2.2073e-41,  4.2728e-40,
-1.0496e-02,  5.4120e-02, -1.6498e-02,
-5.9387e-02,  2.3757e-01, -8.0381e-02,
 2.3739e-02, -1.3715e-01, -3.0906e-02,
-8.5760e-03,  2.4518e-02, -6.9090e-02,
 2.1623e-02,  8.9641e-02,  9.9031e-02,
-1.0052e-02,  4.6506e-02, -1.5756e-01,
 8.5003e-02, -3.6434e-03,  1.3816e-02,
 9.0532e-02,  2.3661e-01,  1.8077e-01,
 2.8120e-02,  4.3753e-02,  2.2981e-02,
 3.5830e-02,  5.7995e-02, -5.6879e-03,
 3.7708e-02, -2.6373e-01,  2.0886e-01,
-4.0632e-02,  1.6891e-01, -6.8996e-02,
-1.1972e-01, -4.3628e-02,  2.0278e-02,
-1.4818e-01,  4.0844e-02,  1.5917e-01,
-4.5684e-02,  1.4075e-01, -2.0784e-02,
-1.1533e-03, -2.7897e-01, -8.8707e-02,
-1.7907e-02,  1.8400e-01,  1.1026e-01,
-2.3183e-03,  6.3875e-02, -4.2394e-03,
 3.2021e-02, -8.8955e-02, -2.2298e-02,
 8.1353e-02,  3.3079e-01, -2.0616e-01,
-3.5802e-02,  4.9804e-02, -9.2712e-02,
-1.5940e-07, -1.6158e-07, -1.5812e-07,
-1.6273e-07, -1.6555e-07, -1.6260e-07,
-1.5867e-07, -1.6192e-07, -1.5975e-07
}
,)"
R"(
{
-1.5080e-02,  1.1294e-01,  7.1187e-02,
 1.1628e-02, -8.4938e-01,  8.5457e-02,
-3.9642e-02, -2.3879e-02,  1.0029e-02,
 2.6648e-40,  9.1590e-41,  3.3285e-40,
-3.3445e-40, -2.5194e-40, -2.0946e-40,
 3.6800e-40, -1.1584e-40,  6.2195e-40,
-1.3560e-41, -8.0151e-41,  4.4048e-40,
-4.1209e-40,  2.7411e-40,  3.2419e-40,
 5.8333e-40,  1.1503e-40, -5.0783e-40,
-5.5301e-02, -2.4971e-02,  4.9251e-02,
-2.5589e-01,  1.6560e-01, -8.0956e-02,
 4.0518e-01,  3.1320e-02, -1.4262e-01,
 1.2250e-02,  5.1989e-02,  3.0706e-03,
-7.9534e-02, -1.9801e-01, -2.7791e-02,
 2.1768e-01,  6.9978e-02, -4.2325e-02,
-1.9165e-02, -2.1179e-02, -2.1558e-02,
 3.6816e-01, -5.2929e-02,  9.5790e-02,
 2.8095e-01, -1.4731e-01,  3.4182e-02,
 2.3702e-02,  4.0764e-02,  3.5767e-02,
-8.4586e-02,  1.9025e-01, -1.6794e-01,
-1.0273e-02,  3.2259e-01, -1.5841e-01,
 2.6794e-01,  5.2084e-02,  1.2761e-02,
-1.1169e-01, -1.7808e-01,  1.1363e-01,
-1.3808e-01, -1.7764e-02, -1.7420e-02,
 1.5840e-02, -2.3405e-01,  7.6361e-03,
-6.6082e-02,  7.9778e-02, -2.0423e-01,
-1.9594e-02, -6.3370e-02,  3.3351e-02,
-2.0396e-40, -3.0207e-40, -3.2364e-40,
 2.3575e-40,  5.8301e-41, -3.7432e-40,
-3.6291e-40,  3.3441e-40,  1.4574e-40,
-4.3792e-40, -2.5814e-40, -3.4986e-41,
-3.4920e-40, -4.4757e-40,  3.2192e-40,
 4.7222e-40, -7.3197e-41, -3.4635e-40,
 5.1495e-02,  7.8843e-02,  4.2243e-02,
-2.1245e-01,  1.9568e-01,  7.9369e-03,
 2.2795e-02,  2.2801e-02,  7.6895e-02,
 3.0044e-01, -1.4041e-01, -2.3677e-02,
-1.1656e-01, -7.5113e-02,  1.0625e-02,
-1.2133e-02,  5.0658e-02, -7.2944e-02,
-3.3652e-02, -2.0452e-01, -4.1048e-02,
 2.8531e-01,  1.2116e-01, -2.1526e-02,
-2.4564e-01, -4.1870e-02, -5.5819e-02,
-2.3157e-01, -2.5594e-02,  1.1154e-01,
 2.1234e-01,  3.2762e-01, -2.9000e-01,
 1.8591e-02, -5.9820e-02, -9.0807e-02,
-3.0027e-01, -1.8370e-01,  1.2086e-02,
 2.1178e-02,  2.9559e-01,  1.2966e-01,
 6.8542e-02,  7.7710e-03, -6.0304e-02,
 3.3019e-03, -1.9135e-02,  9.3227e-03,
-9.9003e-03, -1.0101e-01, -3.3513e-01,
-8.4091e-03, -1.5918e-02, -3.4323e-02,
 3.8770e-40, -2.8639e-40,  4.6953e-40,
 4.2631e-40,  6.2568e-41, -5.3500e-40,
-2.1987e-40,  1.3435e-40,  4.4101e-40,
-3.9973e-40,  6.3046e-40,  1.6046e-40,
 4.4338e-40,  1.6940e-41,  4.1598e-40,
 2.6132e-40, -2.9888e-40, -7.5708e-41,
-1.5991e-02,  8.2749e-02, -6.3776e-02,
-3.2220e-03,  4.1443e-02, -8.1219e-02,
-1.1231e-01,  6.7586e-01, -1.7600e-01,
-4.0371e-02, -7.9044e-02,  1.2451e-01,
 4.1907e-02, -8.8159e-02, -1.1229e-01,
-4.0654e-03, -4.4087e-03,  1.2942e-01,
 9.3318e-03, -6.5085e-02,  1.0165e-02,
-2.8758e-02, -4.9997e-02,  4.6069e-02,
 4.2107e-04,  2.1718e-01,  3.1080e-03,
-9.1277e-03, -2.8568e-02,  1.6202e-02,
-8.2490e-03,  1.2888e-01, -1.3159e-01,
 1.6065e-02,  4.0143e-02,  2.7043e-01,
-3.4809e-02, -8.1302e-03,  6.0786e-02,
 5.1845e-02,  4.6995e-01, -1.0392e-02,
 2.3359e-02, -1.8364e-01, -3.7343e-01,
-8.2996e-02,  9.7724e-02, -6.1012e-02,
 2.8225e-02,  8.8706e-02,  1.3443e-02,
 3.7515e-03,  1.7772e-02,  6.5945e-03,
-7.3847e-12, -7.5629e-12, -6.9337e-12,
-7.6292e-12, -7.8624e-12, -7.2877e-12,
-7.0582e-12, -7.3197e-12, -6.8467e-12,
 1.5445e-11,  2.0754e-11,  2.0524e-11,
 2.1239e-11,  2.5909e-11,  2.5983e-11,
 2.0986e-11,  2.5190e-11,  2.2478e-11,
-4.7164e-02, -2.4754e-02, -1.8256e-02,
 1.0526e-01, -4.6010e-03, -2.2784e-02,
-5.2028e-02, -1.6408e-01,  7.9112e-03,
-8.1863e-02,  4.2772e-02, -9.9446e-04,
-5.5521e-02, -1.1264e-01, -4.5782e-02,
-1.1026e-01,  2.1443e-02, -4.5120e-02,
-1.4141e-02, -2.8116e-03,  2.6990e-02,
-2.0201e-01,  4.3214e-01,  2.9373e-02,
-2.1768e-01, -2.7230e-02,  5.5396e-03,
 5.0196e-02,  1.5506e-01, -5.7328e-02,
 4.8323e-02,  3.8243e-02, -1.3533e-01,
-9.8862e-03, -5.6971e-02, -7.1500e-02,
 1.0272e-01,  7.4686e-02,  7.4732e-02,
 8.3744e-02,  1.5834e-01,  2.9221e-02,
 6.5641e-02,  7.7697e-02,  3.5746e-02,
-1.6614e-01, -2.3128e-01,  4.4691e-02,
 6.3546e-02, -3.8105e-01,  3.4110e-02,
-3.5022e-02, -2.3782e-02,  2.8664e-02,
-3.8813e-41, -2.8626e-40, -9.0218e-41,
 4.1216e-40, -4.4215e-40,  3.1198e-40,
 5.6281e-40,  2.0477e-40,  2.7797e-40,
-4.4903e-40, -6.2574e-41,  4.9971e-40,
 5.0135e-40, -3.1945e-40, -2.4694e-40,
 2.6587e-40, -4.9583e-40, -4.9771e-40,
 3.7139e-02,  5.2936e-04, -2.3658e-02,
-3.6199e-01, -5.1912e-02, -5.1969e-02,
 2.5415e-01,  2.4109e-01,  9.8721e-03,
 5.5061e-02, -4.7469e-02,  3.0045e-02,
 2.1565e-03, -2.3866e-02, -2.3496e-02,
 6.0892e-02, -4.6442e-04, -5.0200e-02,
 5.4971e-02, -1.7234e-02, -3.2759e-03,
 4.8225e-01, -1.1234e-01,  3.8257e-02,
 5.2105e-02, -2.8473e-03, -1.0355e-02,
-9.5654e-03, -1.8751e-01,  1.7079e-02,
 7.0133e-02,  7.6363e-01, -8.7388e-02,
-5.6536e-02, -1.9152e-01, -1.6043e-01,
 2.0359e-01,  7.4214e-02,  3.1970e-02,
-1.8199e-01, -1.9386e-01, -2.5967e-03,
-3.4609e-02,  3.3870e-02,  5.8835e-02,
 8.8220e-02,  9.9265e-02,  7.1240e-03,
-9.1395e-02, -3.1699e-01, -2.9120e-02,
-1.8436e-02, -2.1432e-02, -4.5465e-02,
-3.2013e-40,  3.2019e-40,  4.8747e-41,
 2.6585e-40,  6.1463e-40,  1.4176e-40,
-1.5286e-40,  3.0543e-40,  7.2032e-41,
-6.0758e-40, -3.6200e-40,  1.2123e-40,
 1.3627e-40,  3.2983e-40,  3.6171e-40,
-4.2148e-40,  1.1102e-40,  3.2714e-40,
-3.4763e-02, -3.1632e-02,  3.0044e-02,
-2.0935e-01,  1.3533e-01, -9.1607e-03,
-1.5931e-01,  1.0771e-01, -6.6518e-02,
 2.4399e-02,  2.2923e-03,  5.1575e-02,
-1.4154e-01, -1.0013e-02, -7.5696e-02,
 1.0849e-01,  1.2575e-01, -7.3161e-02,
-1.5217e-02, -2.7659e-02, -3.1401e-02,
 3.4960e-01,  7.2390e-02,  2.0722e-02,
 3.9440e-01,  9.1821e-04,  1.7842e-02,
-1.5670e-02,  5.3020e-02,  6.0536e-02,
-1.8853e-01,  2.7532e-01, -1.9681e-01,
 8.3258e-02,  9.4285e-02, -1.2695e-01,
 2.7593e-01,  1.1456e-01,  1.6048e-02,
-5.1675e-01,  1.4727e-01,  7.5170e-02,
-6.9143e-02, -9.2948e-02,  3.4687e-02,
 1.4128e-02, -7.9962e-02,  8.0446e-02,
 3.7011e-02, -1.3400e-01, -2.0725e-02,
-6.4981e-03,  7.0724e-02,  6.6167e-02,
-4.5940e-41,  2.5437e-40, -3.3111e-40,
 5.9661e-40,  6.2521e-40,  5.6418e-40,
 1.9187e-40, -5.8872e-40,  5.5747e-40,
-1.6402e-11, -2.2097e-11, -1.7224e-11,
-2.2755e-11, -2.9977e-11, -2.1231e-11,
-1.3688e-11, -1.7479e-11, -1.3081e-11,
 6.4790e-03, -3.8464e-03, -1.0008e-02,
-2.6001e-02, -7.9483e-02,  3.3711e-02,
 2.6659e-03, -3.2634e-02,  1.0767e-02,
 4.9939e-03,  1.4064e-02, -3.4294e-02,
 4.8529e-02,  6.3386e-01, -3.6805e-02,
-1.3703e-01,  2.5878e-02, -4.8617e-02,
 3.2186e-02,  6.6382e-02,  1.9305e-02,
 7.0196e-02, -1.6892e-01, -2.8980e-02,
 9.7762e-02,  9.7998e-03, -5.1620e-03,
 5.0753e-02, -4.5071e-03, -3.9836e-02,
-6.0381e-02, -9.2016e-02,  9.5433e-02,
-1.0045e-02,  8.7955e-03,  4.9429e-02,
-1.8363e-02, -1.1912e-01,  9.7347e-03,
-1.5657e-01, -2.1035e-01, -4.9737e-02,
-3.0025e-02, -6.4959e-02, -5.6107e-02,
 3.2927e-40,  5.7263e-40,  6.2889e-40,
-6.0716e-39,  5.3050e-41, -1.7152e-40,
-3.2493e-38, -1.5841e-40, -1.9343e-40,
 4.9763e-40,  5.5142e-40, -4.3462e-40,
-2.2649e-40,  1.4321e-40, -2.6779e-40,
 2.3072e-41,  5.4080e-40, -6.4200e-41,
 2.2827e-40, -5.4515e-41, -4.1768e-40,
 3.9033e-40,  6.1988e-41,  5.9877e-40,
-4.3355e-41, -5.1088e-40,  5.9845e-40,
-4.8238e-40, -1.8586e-40,  4.8699e-40,
-9.7225e-41,  4.3387e-40, -4.3683e-40,
-7.9278e-41, -5.3614e-40,  2.1911e-40,
-3.3982e-40, -5.3335e-40,  3.8540e-40,
 1.9051e-40, -2.0840e-40,  2.2868e-40,
-3.5020e-40, -3.4276e-40,  2.7395e-42,
 3.9197e-40,  6.1843e-40, -1.5888e-40,
 4.3516e-40, -6.1852e-40, -5.3692e-40,
-4.3268e-40,  3.5154e-40,  3.4477e-40,
-4.8414e-40,  2.2647e-40, -2.5591e-40,
 4.6326e-40, -3.0462e-40,  4.7817e-40,
-4.9853e-40, -5.3425e-40, -2.9848e-40,
-1.3329e-07, -1.3784e-07, -1.3049e-07,
-1.3376e-07, -1.3905e-07, -1.3204e-07,
-1.2479e-07, -1.2994e-07, -1.2410e-07
}
,)"
R"(
{
-2.5964e-02,  2.9670e-02,  1.2100e-01,
-3.0371e-02, -1.5277e-02, -1.8589e-01,
-1.8650e-02, -1.2852e-01, -6.6297e-02,
 9.7934e-04, -5.1835e-02, -1.0278e-03,
-1.2336e-02,  2.2130e-01, -1.2373e-01,
-2.3451e-02,  3.4217e-02, -1.0118e-02,
-3.0558e-01, -8.5390e-02, -1.4360e-02,
 1.2473e-01, -1.7005e-02, -3.6816e-02,
-8.9125e-02, -6.1400e-02, -2.0623e-02,
 1.3736e-02,  1.2441e-02, -4.3491e-02,
 6.4806e-02,  3.7012e-01,  3.8064e-02,
-1.3731e-02, -2.4859e-01, -2.5450e-01,
-6.5111e-03, -1.4271e-01, -5.0481e-02,
 5.3240e-02, -3.4843e-02, -2.2703e-02,
 3.7414e-02,  1.0334e-01, -7.2237e-02,
 1.4216e-02,  3.4231e-02, -2.0890e-02,
 2.7879e-02,  1.3717e-01,  4.5864e-03,
 3.0460e-03, -1.1734e-01,  4.4439e-02,
 6.4825e-03,  1.6324e-02,  1.4928e-02,
-8.8420e-02, -1.0779e-01, -9.0653e-02,
 3.1086e-02, -2.9067e-02, -8.8488e-02,
-1.6779e-40, -6.3646e-41, -6.2486e-40,
 2.3154e-40,  2.8049e-40,  3.7718e-40,
-3.3950e-40, -3.1501e-40,  5.8709e-40,
 2.1435e-02, -4.3732e-01,  1.5520e-02,
 3.4080e-02,  1.9912e-01, -8.1413e-02,
-3.2816e-02,  5.7844e-02,  8.9258e-03,
-1.1662e-02, -1.1721e-02,  4.3033e-02,
 5.2135e-02, -2.2503e-01,  2.3941e-01,
 3.8400e-02,  1.8075e-01, -1.4776e-01,
 2.6784e-01,  2.2817e-01, -3.0553e-03,
-6.7998e-02, -1.2050e-01,  1.4714e-02,
 2.4045e-02, -1.4329e-02, -1.6705e-02,
-1.1421e-02,  4.2139e-02,  4.2944e-02,
 1.8809e-02, -2.5221e-01,  9.7562e-02,
-4.1600e-02,  4.0069e-03,  7.5290e-02,
-2.0092e-02,  2.3537e-01,  2.4356e-02,
 3.1957e-02, -4.8573e-02,  2.9379e-02,
 6.4562e-03, -1.1527e-01, -9.1223e-02,
-2.3432e-02,  5.2881e-02, -7.3239e-02,
-3.7048e-02, -2.1481e-01,  5.9801e-05,
-4.2646e-02, -1.8366e-02, -1.0681e-01,
-1.3366e-01, -1.7123e-01, -3.5629e-02,
 1.1216e-01,  1.1479e-01,  9.5297e-02,
 2.4728e-02, -7.3135e-03, -3.4373e-02,
-2.3917e-40, -4.1869e-41,  3.7775e-41,
 2.8931e-40, -9.4850e-41,  2.5694e-40,
 3.3549e-40, -2.4334e-40, -5.5933e-41,
-2.0900e-02,  2.1203e-02, -4.7169e-02,
 2.3632e-02, -7.1148e-01,  4.9722e-02,
-7.8963e-03,  5.0689e-02,  2.2619e-02,
-4.7364e-03,  3.2037e-02,  1.1004e-02,
-4.3001e-03,  2.5245e-01,  5.9112e-02,
 2.8932e-02, -1.1267e-01, -2.3739e-01,
-6.5379e-02,  5.2462e-03, -1.6807e-02,
 1.0960e-01,  1.7943e-01, -6.3043e-03,
 9.3102e-02,  7.3103e-02,  2.5259e-02,
 5.6835e-02,  4.0467e-02,  2.5447e-03,
 9.4599e-02,  2.5222e-01,  6.9855e-02,
 4.4758e-02,  1.8073e-01,  1.5075e-01,
 2.0329e-02, -4.9412e-02,  2.0663e-02,
-7.1648e-03,  1.4986e-01,  2.1212e-01,
 2.7657e-02, -6.8660e-02,  1.7321e-02,
 1.0629e-02, -1.0722e-02,  2.8247e-02,
-1.1303e-02,  1.0076e-01, -4.0592e-01,
 2.6744e-02,  7.3650e-02,  5.7966e-02,
 2.8122e-02, -7.5961e-02, -9.4797e-03,
-1.3010e-01, -5.4184e-01, -1.3619e-01,
-1.8661e-03, -1.4357e-01,  7.9520e-03,
-1.3538e-09, -1.6580e-09, -1.7289e-09,
-1.2386e-09, -1.5132e-09, -1.5987e-09,
-1.1157e-09, -1.3420e-09, -1.4090e-09,
 1.5441e-02, -1.8142e-01, -8.6802e-02,
-4.0983e-02,  2.4351e-01, -5.8181e-02,
-2.9568e-02,  3.9561e-03,  3.4181e-02,
-2.9210e-02,  2.5403e-02,  9.1331e-02,
 2.3621e-02,  2.3954e-01,  5.2487e-02,
 1.6509e-02, -6.2728e-02,  1.3448e-02,
 1.2855e-01,  1.1892e-02, -1.3356e-02,
 1.0810e-01,  1.6760e-01, -3.2040e-02,
 6.2209e-02,  4.0682e-02,  3.9772e-02,
-6.1711e-03,  5.0588e-02, -1.0811e-01,
 1.5744e-02,  1.6091e-01, -6.1739e-02,
-5.6717e-02, -1.0657e-02, -3.7943e-02,
-4.0595e-02,  8.0149e-02,  2.0216e-02,
 3.8838e-02, -6.3586e-01,  2.3785e-01,
-1.0472e-02,  6.3899e-02, -8.2184e-02,
-1.9137e-02,  8.1163e-02,  6.7065e-02,
-2.2377e-03,  1.1860e-01,  3.4122e-02,
 1.0501e-02,  2.9851e-02,  7.5841e-02,
 5.8970e-02, -1.2188e-01,  7.7982e-02,
-2.6516e-02, -4.1289e-01,  2.1471e-02,
 3.3957e-02,  3.5762e-02, -5.7857e-02,
-2.7357e-30, -3.4780e-30, -3.0306e-30,
-1.5188e-30, -1.9888e-30, -1.8755e-30,
-7.7431e-31, -9.7571e-31, -9.7402e-31,
-1.8497e-02, -2.4554e-02,  1.4428e-01,
 1.4217e-02, -2.3647e-01,  8.4097e-02,
-1.0251e-02, -4.2137e-03,  6.0831e-03,
 1.7742e-03,  2.1487e-02,  3.3147e-02,
-1.0971e-02,  3.0162e-01,  5.2391e-02,
 1.8341e-02, -1.3390e-01,  9.4303e-02,
-1.5685e-01,  9.8434e-02, -1.2502e-03,
 3.1370e-01, -2.8879e-02,  2.6313e-03,
 1.7548e-02,  6.6741e-03, -1.7681e-03,
 5.2062e-02,  6.6914e-02,  7.5256e-03,
 2.4966e-02,  2.8081e-01,  2.9815e-02,
 2.2375e-02,  1.4257e-03, -7.4702e-02,
 1.5372e-02,  3.9587e-02,  4.6909e-02,
-2.2911e-02, -1.4568e-01, -3.8964e-01,
 2.2850e-02, -4.2297e-02,  6.5736e-02,
-6.9905e-03, -6.3972e-02, -1.8430e-01,
 4.4453e-03,  2.0687e-01,  3.0032e-01,
 1.7243e-02,  9.8548e-03, -9.7476e-02,
-7.9682e-04, -2.1199e-01, -4.3461e-02,
-4.2929e-02, -2.8227e-01,  2.8997e-02,
-1.8741e-03,  1.1166e-02,  1.8381e-03,
-5.6725e-16, -1.0368e-15, -1.1480e-15,
-5.5537e-16, -9.9929e-16, -1.1499e-15,
-3.8787e-16, -6.4019e-16, -7.7595e-16,
 4.4505e-02,  8.8803e-02,  1.1384e-02,
-3.9434e-02,  1.9319e-01, -1.2016e-02,
-4.6072e-02,  1.1769e-01,  7.4816e-03,
-3.7856e-02, -1.7147e-02,  1.5984e-01,
-2.6459e-02,  1.7469e-01,  1.2584e-01,
 1.6387e-02,  1.7370e-01, -1.7350e-01,
-3.0008e-01,  2.1485e-01, -5.4302e-02,
 5.7724e-02,  3.2168e-01, -2.5261e-02,
 6.9277e-02,  7.5035e-02,  6.3485e-02,
-1.1688e-01,  2.6068e-02, -1.3490e-01,
-1.6085e-01,  1.9409e-01,  1.1434e-01,
-7.3819e-02, -7.7880e-02,  7.3699e-03,
-9.9972e-02,  1.3554e-01,  2.1656e-02,
-8.8303e-02,  5.4435e-01, -4.0582e-02,
-3.4805e-02, -1.5291e-01, -3.6917e-02,
-3.4377e-02, -3.3086e-02, -9.5097e-02,
-7.4538e-03,  2.2545e-01, -2.6380e-02,
 1.4440e-02,  1.3205e-01,  1.6164e-01,
 9.2164e-02, -8.4307e-02,  7.8922e-02,
 1.2519e-01, -6.1809e-01, -1.0895e-01,
 6.2744e-02, -4.4951e-02, -3.2548e-02,
-2.5422e-21, -6.3849e-21, -9.5560e-21,
-1.9248e-21, -4.7107e-21, -6.4244e-21,
-1.4638e-21, -3.1947e-21, -3.7663e-21,
-8.6113e-03, -7.0987e-02,  5.8265e-02,
-1.3148e-02,  5.6371e-01,  5.0580e-02,
 1.1741e-02, -3.5614e-02, -6.1265e-02,
 1.4758e-03,  3.3349e-02, -1.0867e-02,
-4.0234e-02,  1.9894e-01,  1.3972e-01,
-1.9167e-02, -4.1723e-02, -1.9982e-01,
-3.0756e-01,  2.6284e-02, -1.9058e-02,
-7.9349e-04,  1.2644e-01,  2.9567e-02,
-3.9274e-02,  1.1030e-02, -9.4885e-03,
 1.3541e-02,  1.7044e-01,  8.9626e-02,
 6.6814e-02,  2.6430e-01,  1.7409e-01,
-6.1034e-04,  1.7569e-02,  1.3090e-01,
-4.1941e-03,  8.9599e-02, -3.3684e-02,
-1.1310e-02, -4.3731e-01,  5.7177e-02,
-4.5718e-04,  1.0175e-01,  4.1211e-02,
 2.9756e-02, -1.1601e-01, -7.3171e-02,
 2.7939e-02,  2.1334e-01, -4.0210e-01,
-8.6847e-03,  8.1829e-02,  4.4225e-02,
-1.1411e-01, -1.7697e-01, -5.8087e-02,
 7.9613e-02, -4.2814e-01, -1.0814e-01,
-3.0610e-02,  1.1342e-03, -2.2322e-03,
-1.1254e-10, -1.4207e-10, -1.5402e-10,
-9.9123e-11, -1.2394e-10, -1.3338e-10,
-8.8840e-11, -1.0857e-10, -1.1463e-10,
 3.0283e-02, -5.6191e-02, -1.0447e-01,
-1.4578e-02, -2.8745e-01,  1.9089e-01,
-2.7251e-02,  9.8069e-02, -1.4580e-02,
-3.0276e-02,  1.4366e-02,  2.6363e-02,
-8.4962e-02,  7.8998e-02, -4.7717e-02,
-3.2004e-02, -2.1579e-02,  1.1247e-02,
 1.3895e-01, -3.3900e-01,  7.7998e-03,
 2.4769e-01, -1.8506e-01, -2.3116e-03,
 3.1361e-02, -1.1718e-02, -1.8286e-02,
-1.3020e-01,  1.4334e-01, -5.5700e-02,
-3.5386e-02,  1.0992e-01, -8.0235e-02,
-5.8978e-03,  7.7039e-02, -7.4619e-02,
-8.1603e-02,  1.2982e-01, -7.3193e-02,
-6.1469e-02,  1.7131e-01,  4.0255e-01,
-6.4582e-03, -8.2741e-02, -2.2220e-02,
 1.6876e-02, -3.2590e-02,  5.5645e-02,
 2.5231e-02,  2.9984e-01, -3.6995e-02,
 9.3322e-03,  2.0758e-01, -2.1986e-02,
-4.9568e-02,  2.1857e-03,  8.6127e-02,
 8.6593e-02, -5.8134e-01,  3.4507e-01,
 4.8855e-02, -1.0506e-01,  4.1584e-02,
 2.5428e-40, -4.4558e-40, -2.2090e-40,
-2.9727e-40, -4.8454e-40,  3.0397e-40,
 1.1696e-40, -3.3028e-40, -2.2959e-40
}
};)"
R"(
__constant float biasL[8][8] = 
{
{
-3.1869e-08, -3.8279e-01, -6.3693e-05, -5.9054e-02,  9.3774e-04, -2.9944e-02, -1.1156e-03, -7.5635e-02
}
,
{
-1.7701e-01, -1.3417e-06, -3.0706e-40, -1.9022e-06, -1.2965e-02, -6.6444e-40,  1.4699e-02,  2.6082e-02
}
,
{
-3.7577e-07,  4.4550e-03, -8.1266e-04,  3.2408e-01, -1.1321e-07, -1.8907e-23, -1.9770e-25, -3.2394e-02
}
,
{
-2.1525e-14, -1.4130e-02, -1.9410e-02, -1.8703e-02, -2.9177e-02, -4.0635e-02,  7.8097e-02, -1.1643e-01
}
,
{
-2.6309e-02, -2.2238e-02,  6.8700e-03, -1.7973e-02, -1.0893e-02, -1.1888e-02, -4.9598e-03, -6.3663e-06
}
,
{
-1.2406e-03, -2.4901e-12, -9.7265e-07,  6.3490e-03,  1.3495e-01, -3.8411e-03, -6.6630e-03, -7.3614e-03
}
,
{
-2.7729e-03, -4.8174e-03, -6.3012e-03,  2.0491e-01, -2.0110e-03, -3.0974e-03,  5.1407e-01, -3.5016e-08
}
,
{
0.0324, 0.0140, 0.6750, 0.2661, 0.3646, 0.3591, 0.5597, 0.0816
}
};

__constant float kernelsL10[4 * 8] = 
{
 0.0882,  0.0422,
 0.3775,  0.4754,
-0.3209, -0.4870,
-0.0384,  0.0530,
 0.1034,  0.0173,
 0.5011,  0.3900,
 0.3621, -0.1645,
-0.1304,  0.0013,
 0.2230,  0.3026,
 0.1618, -0.4514,
-0.2097,  0.1894,
-0.0326,  0.1434,
 0.2421,  0.3363,
-0.0938,  0.3156,
 0.1137, -0.2165,
 0.2273, -0.1284
};)") + kernelFunction
,
std::string(
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
-2.0676e-02,  6.7641e-03,  2.8287e-01,
 2.5576e-01,  1.9765e-01, -2.4700e-01,
 3.5056e-01,  2.9306e-01, -2.2245e-01,
 8.4706e-02, -2.9455e-01, -5.5831e-02,
-8.4635e-02, -9.6835e-02,  3.1208e-01,
 1.7690e-01,  2.7624e-02,  5.1954e-02,
-5.3869e-01,  7.2934e-02, -1.7662e-03,
-3.1402e-02,  3.1700e-01,  1.4965e-01,
 3.8569e-02,  5.5025e-03, -6.6555e-03,
-4.2049e-38, -4.1971e-38, -4.1488e-38,
-4.2855e-38, -4.2871e-38, -4.2363e-38,
-4.1861e-38, -4.1974e-38, -4.1677e-38,
 1.8451e-01, -5.4584e-02,  1.4494e-01,
 1.3433e-01,  1.0073e-01,  2.6371e-01,
 6.1261e-02,  2.2116e-01,  2.0074e-01,
 5.9669e-02, -3.9168e-02,  2.1674e-01,
-2.9132e-01,  3.0285e-03,  1.2625e-01,
-4.3415e-02,  1.8663e-01, -1.6554e-01,
 1.0102e-01,  6.3466e-02,  1.5225e-01,
 2.1692e-01,  1.9860e-01, -7.0456e-02,
-1.6406e-03, -2.7834e-01, -3.5449e-01,
-3.0140e-01, -4.2348e-01, -5.8263e-01,
 2.3140e-01, -2.6843e-01, -1.1069e-01,
-9.1484e-02,  1.1486e-02,  5.6396e-02
};

__constant float biasL1[8] = 
{
-9.0964e-02,  2.1136e-01, -1.2011e-02, -4.5657e-38, -1.4443e-01, 1.8968e-01, -2.9027e-02,  1.6199e-01
};
)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
 4.4561e-02,  4.3527e-01, -8.9737e-02,
-4.9011e-03,  1.4879e-01, -8.2210e-02,
-1.7593e-02,  4.9294e-02,  1.8058e-01,
-3.3827e-02, -7.9055e-02,  2.6982e-01,
-5.2485e-02, -4.2046e-01, -5.6838e-02,
 1.0919e-01, -7.3141e-02,  9.4797e-02,
 6.2764e-02,  2.5475e-01,  1.3705e-01,
 2.0997e-01,  7.3360e-01,  2.0801e-01,
-1.1500e-01,  3.1245e-01,  6.7457e-01,
-5.1481e-39, -5.1520e-39, -4.9367e-39,
-5.1383e-39, -5.1642e-39, -4.9479e-39,
-5.1323e-39, -5.1859e-39, -4.9547e-39,
 1.3849e-01,  1.1564e-01, -1.8175e-01,
-5.5355e-03, -1.5117e-01, -2.4654e-01,
 8.1590e-03, -1.1681e-01,  3.4700e-05,
-2.5950e-01, -1.4182e-01,  3.1814e-01,
 1.7662e-01,  1.8420e-01, -1.5181e-01,
 7.6233e-02, -7.8372e-02, -3.1968e-01,
-4.5770e-01,  4.1562e-02,  1.3721e-01,
-5.8444e-02,  3.3148e-02, -2.3370e-01,
 1.5374e-01, -1.1162e-01, -7.4099e-03,
-1.5716e-01, -1.8356e-01,  2.1114e-02,
-3.2233e-01,  2.1064e-02,  2.7019e-01,
-1.3702e-01,  2.6969e-01,  2.1033e-01,
 8.9027e-02, -7.9969e-02,  1.0096e-01,
 6.6773e-02,  3.9558e-02, -7.4944e-02,
-5.9789e-02,  1.2265e-01,  3.3873e-02,
-9.7157e-03,  9.2906e-02,  6.0300e-02,
-2.2104e-03,  6.8198e-02, -1.2931e-01,
 8.9288e-02, -1.2554e-01, -4.3270e-02,
 1.0660e-01,  1.1609e-02, -1.2415e-01,
 2.6372e-02, -3.6311e-02,  1.5625e-01,
-7.9595e-02, -3.3662e-01, -4.0760e-01,
-2.9566e-39, -2.8760e-39, -2.8816e-39,
-2.9566e-39, -2.8964e-39, -2.9115e-39,
-2.9566e-39, -2.9179e-39, -2.9130e-39,
 7.9255e-02,  9.4548e-02,  8.8155e-02,
-2.8163e-02,  1.2428e-01, -6.4973e-03,
 7.7875e-02,  7.4765e-02, -5.2405e-02,
-1.4886e-02, -7.1499e-02, -7.0719e-02,
 9.7562e-02,  9.0948e-02, -5.6588e-02,
-1.2872e-02, -6.6390e-02, -6.4147e-02,
 9.8262e-02, -2.4215e-01, -1.7051e-01,
 1.8096e-01,  1.8106e-01,  1.3108e-01,
 2.0649e-01,  1.2242e-01,  3.7225e-02,
-2.5125e-01, -1.0073e-01,  4.5330e-01,
 1.8588e-01, -2.6809e-01, -1.5709e-01,
 4.7668e-01, -2.4208e-01, -6.6012e-01,
 1.3561e-01,  5.4109e-02,  6.1899e-02,
-1.9605e-02,  1.1349e-01,  3.5781e-02,
 3.5513e-03,  3.1212e-02, -6.0399e-02,
 5.9258e-02, -1.8175e-02,  7.3714e-02,
 2.0052e-02,  4.3245e-02, -5.0879e-03,
-1.1082e-02, -1.0753e-01, -1.7896e-03,
 2.9139e-02,  2.2747e-01, -6.4075e-02,
 7.3097e-02,  1.5703e-01, -5.3815e-01,
 1.0620e-01, -1.1386e-01,  1.7103e-01,
-3.8728e-39, -3.8299e-39, -3.8320e-39,
-3.9065e-39, -3.8445e-39, -3.8135e-39,
-3.8838e-39, -3.8114e-39, -3.8255e-39,
 2.3253e-02,  6.9893e-02,  1.4774e-01,
 9.6087e-02,  2.3102e-03, -3.4449e-02,
 2.6819e-02,  1.0254e-01, -2.8200e-02,
 3.9553e-02,  4.7191e-05, -5.5558e-02,
 4.1641e-02,  5.8706e-02, -1.0337e-01,
 1.1291e-01,  5.9622e-02,  7.0677e-02,
-2.5162e-01,  7.6659e-02,  1.7245e-01,
-5.8522e-02,  1.4365e-01,  2.1189e-01,
-2.8897e-02, -5.7365e-02,  1.4232e-01,
 1.7854e-02,  1.7404e-03, -8.7356e-03,
-6.0777e-02, -6.2687e-02, -1.1500e-02,
-1.6468e-01, -2.5058e-01, -1.2798e-01,
 2.3193e-02,  1.7209e-01,  1.6687e-01,
-3.4483e-02, -1.6846e-02,  2.5930e-02,
 1.4410e-01,  4.2932e-02, -5.0149e-03,
 4.7269e-02,  1.1276e-01, -9.2701e-03,
 1.5323e-02,  1.3552e-02,  9.0256e-02,
-8.9393e-03,  7.0903e-02, -6.9379e-02,
 1.8645e-01,  1.0543e-01, -1.5590e-01,
 2.1056e-01,  1.1051e-01, -1.5514e-01,
-7.0484e-02, -1.5153e-01, -5.0873e-01,
 3.2730e-39,  3.2358e-39,  3.1222e-39,
 3.2642e-39,  3.2358e-39,  3.0921e-39,
 3.2730e-39,  3.2358e-39,  3.0899e-39,
 1.2225e-02,  1.2386e-01,  6.7712e-02,
 3.1263e-02,  1.3617e-01,  1.5352e-01,
 2.3405e-02,  8.5466e-02,  8.7303e-02,
-2.0372e-02,  8.3465e-02, -7.4233e-02,
 1.2269e-01,  8.4046e-02, -3.6869e-02,
 1.0242e-01,  7.3218e-02, -1.1496e-01,
-1.4539e-01, -2.3923e-01, -2.2818e-01,
-3.2368e-02, -7.4360e-02,  2.3493e-02,
 1.7004e-01,  6.2924e-02,  8.9327e-02,
-1.1449e-01, -1.4973e-03, -7.0451e-03,
-9.3205e-02, -1.0312e-01,  4.6503e-02,
-2.2148e-01, -1.8111e-01, -1.1992e-01,
 9.8140e-02,  9.9823e-02, -2.0282e-02,
-8.1973e-02,  1.4255e-01, -5.2392e-02,
 8.0350e-03, -4.8299e-02, -7.7908e-02,
 4.2383e-02,  3.0707e-02,  2.8560e-02,
 1.0437e-01,  6.1290e-02, -9.7796e-02,
-1.7125e-02, -1.3572e-01, -1.5345e-01,
-1.3292e-01,  2.9477e-02,  6.8032e-02,
 1.5741e-01,  4.0258e-01,  2.5838e-01,
 1.3948e-01,  3.5713e-01, -3.9825e-01,
-1.9224e-39, -2.4076e-39, -2.4529e-39,
-1.9181e-39, -1.9894e-39, -4.0240e-39,
-1.9335e-39, -2.3920e-39, -4.0147e-39,
-2.1714e-02, -3.5299e-02, -7.5803e-03,
-2.4087e-02,  7.5265e-02,  7.6697e-02,
 4.5309e-02,  8.9529e-02,  7.6510e-03,
 1.0813e-02,  3.1294e-02, -2.5907e-02,
 1.1962e-02, -6.8664e-03, -1.4084e-01,
 7.7013e-02, -1.2305e-01, -6.7800e-02,
-9.7392e-02,  4.4082e-02,  1.4473e-01,
 4.9436e-02,  2.8859e-01,  2.8252e-01,
-3.5828e-02, -7.5616e-02,  2.4875e-01,
-6.7684e-02,  1.1290e-01,  4.2827e-02,
-1.0860e-01,  1.2952e-01,  5.9784e-01,
-3.5402e-01, -3.9558e-02, -6.0775e-01,
-1.2854e-02,  1.5240e-01,  1.4115e-01,
-2.8134e-02, -1.2939e-02, -2.6203e-02,
 1.1300e-01,  1.4481e-01, -5.1454e-02,
 1.2688e-01,  2.8536e-02,  9.4877e-02,
 9.6033e-02, -1.3901e-02,  6.0035e-02,
-1.1249e-01,  4.3971e-02, -1.0918e-01,
 8.2500e-02,  2.1413e-01,  3.9015e-02,
 1.8361e-01,  2.5271e-01, -2.2794e-01,
-8.1195e-02, -1.2269e-01, -2.6097e-01,
 7.6827e-39,  7.7882e-39,  7.6893e-39,
 7.7006e-39,  7.7857e-39,  7.7384e-39,
 7.6985e-39,  7.7712e-39,  7.7399e-39,
 1.4458e-02,  1.0801e-01,  1.5906e-01,
-1.4676e-02,  1.3699e-01,  9.2460e-02,
-3.6479e-02,  1.4529e-01, -2.8681e-02,
-3.3251e-02, -7.3096e-02, -1.4330e-01,
 5.7009e-02, -3.1905e-02, -1.2035e-01,
 1.1838e-01,  5.7011e-02,  2.0800e-02,
-1.1567e-02, -2.2125e-01, -9.3953e-02,
-7.5378e-02, -1.2069e-01,  1.3217e-01,
-7.7357e-02, -1.3171e-01,  1.2776e-01,
-1.1397e-01, -3.5183e-02,  2.2994e-02,
-6.5101e-02, -1.5019e-01, -2.7451e-02,
-2.4260e-01, -1.3543e-01, -1.9889e-02,
-1.9798e-39, -3.5282e-40, -1.9216e-39,
-1.9140e-39, -1.9370e-39, -1.9943e-39,
-1.8623e-39, -1.8665e-39, -1.9320e-39,
-4.8850e-39, -5.0283e-39, -4.9987e-39,
-5.0868e-39, -5.0814e-39, -5.0779e-39,
-5.2489e-39, -5.1086e-39, -5.1234e-39,
-2.9120e-39, -3.0278e-39, -2.9633e-39,
 1.3186e-39,  6.0555e-39,  6.0419e-39,
-5.5922e-39, -8.5992e-40, -2.8529e-39,
-3.4668e-39, -3.5127e-39, -3.4668e-39,
-3.2831e-39, -3.4668e-39, -3.6734e-39,
-3.2142e-39, -3.2831e-39, -3.5816e-39,
 1.3445e-39,  1.3621e-39,  1.3375e-39,
 1.4539e-39, -2.2695e-40,  1.4522e-39,
 1.3563e-39,  1.3339e-39,  1.3001e-39,
-4.4670e-39, -4.4026e-39, -4.3159e-39,
-4.5047e-39, -4.3505e-39, -2.7259e-39,
-4.5265e-39, -4.4721e-39, -4.4990e-39,
-1.9864e-39, -4.1379e-39, -3.7189e-39,
 5.2465e-39,  2.5220e-39,  1.5639e-39,
-3.9760e-39, -5.7033e-39, -4.0978e-39,
-6.3745e-40, -4.7511e-39,  2.3456e-39,
-1.5164e-39,  5.0431e-39,  5.1197e-39,
 8.7052e-40,  1.4947e-39, -1.1546e-39,
 5.3140e-02,  1.0281e-01,  1.4767e-01,
-6.1530e-02, -9.4166e-02,  4.8671e-02,
 5.6787e-03, -1.4551e-01,  1.5614e-02,
-3.4826e-02, -5.1148e-02,  9.7079e-02,
-1.3603e-02, -1.2249e-01, -1.9330e-02,
-6.8184e-02, -1.4344e-01, -9.4023e-03,
-7.4629e-02,  3.9634e-02,  1.3445e-01,
 4.2153e-02,  7.1129e-01,  2.8703e-02,
 7.8247e-02,  7.2210e-01, -6.6198e-01,
-6.1010e-39, -6.2892e-39, -6.4008e-39,
-6.0825e-39, -6.3221e-39, -6.3883e-39,
-1.4962e-39, -1.1702e-39, -1.2143e-39,
 5.5512e-02, -2.1522e-02,  1.0866e-01,
-9.2812e-02, -3.5119e-02,  1.1396e-01,
-1.3922e-01,  6.7287e-02, -5.5626e-02,
-2.0492e-01,  8.1441e-02, -1.3513e-01,
 4.7447e-02,  2.0081e-01, -3.1249e-01,
-1.8546e-02,  2.0680e-01,  7.3979e-02,
 8.8928e-02, -4.3606e-01, -8.4823e-02,
-5.6133e-02,  3.5132e-01,  1.8633e-01,
-4.3855e-03,  5.4869e-02,  1.1658e-01,
 1.7423e-01, -5.3107e-02,  2.2925e-02,
-1.7622e-01,  4.4453e-02,  2.8131e-02,
 2.6863e-01, -2.9085e-01, -1.5098e-01
}
,)"
R"(
{
-2.4230e-40,  5.4425e-39,  3.4517e-39,
-1.9803e-39, -1.5207e-39, -3.5630e-39,
-4.9409e-39, -2.9280e-39,  7.7966e-40,
 2.4867e-39, -2.1848e-39,  3.2524e-39,
-6.2860e-39,  4.0411e-39, -3.6956e-39,
-3.3384e-39, -1.0908e-39,  5.4261e-39,
-3.6691e-40,  9.4949e-40, -1.7279e-39,
-1.0644e-39, -2.1371e-39, -2.5125e-39,
 2.9368e-39, -5.3820e-39, -3.9771e-40,
-1.4703e-39, -3.6960e-39, -4.4161e-39,
 8.2800e-40, -4.9175e-39,  3.1868e-39,
 5.5703e-39, -3.0263e-39, -1.6991e-39,
 5.2691e-39,  4.8127e-39,  4.1346e-39,
-1.3013e-39, -1.7101e-39, -3.5467e-39,
 1.1496e-39,  2.0938e-39, -4.2970e-39,
-5.5314e-39,  6.4852e-40, -5.0870e-39,
 3.9377e-39, -4.1683e-39, -3.5404e-40,
-3.6188e-39,  5.4657e-39,  2.1279e-39,
 3.4090e-40,  2.4425e-40,  9.3423e-41,
-2.3450e-39,  3.1518e-40,  4.3061e-40,
-2.6175e-39, -2.4696e-39, -2.3755e-39,
 2.2764e-39, -4.4934e-39,  8.5722e-40,
 5.1798e-39,  2.7072e-39,  5.3750e-39,
 5.4335e-40,  3.8556e-39, -3.4799e-39,
-4.8963e-39, -1.1413e-39, -5.3918e-40,
 6.1843e-39, -1.8521e-39, -1.3450e-39,
-2.0906e-39, -3.2544e-39, -2.8205e-39,
 5.3550e-39, -3.0202e-39, -3.4181e-39,
-3.0043e-39, -3.2900e-39, -3.2915e-39,
 6.1849e-39, -3.3421e-39, -3.3995e-39,
-4.8657e-39, -4.7034e-39, -4.7467e-39,
-4.6555e-39, -4.6045e-39, -4.6954e-39,
-4.8886e-39, -4.7333e-39, -4.7805e-39,
-2.0900e-39, -1.9429e-39, -2.0572e-39,
-2.0270e-39, -1.9074e-39, -1.9275e-39,
-2.1243e-39, -2.1134e-39, -2.1539e-39,
-4.4175e-39, -4.6412e-39, -4.6582e-39,
-4.6364e-39, -4.8757e-39, -4.6795e-39,
-4.4571e-39, -4.5038e-39, -4.4570e-39,
-3.2662e-39, -3.1163e-39, -3.2050e-39,
-3.2098e-39, -3.0887e-39, -3.1635e-39,
-3.3183e-39, -3.1411e-39, -3.2824e-39,
 8.6839e-40,  5.7318e-39,  1.8373e-40,
 4.6732e-39, -4.5549e-41,  1.2817e-39,
 3.7642e-41, -6.2591e-39, -5.0492e-39,
 5.0057e-39,  6.0612e-39,  2.0220e-39,
 3.7436e-39,  4.8326e-39,  3.1353e-39,
 3.5289e-39,  4.7177e-39,  6.2666e-39,
-1.4963e-01, -8.0360e-02, -7.9054e-02,
-1.3731e-01,  5.0766e-02,  6.9673e-02,
 3.2213e-02,  3.3250e-02,  1.3170e-01,
-2.9718e-02, -2.6931e-02,  1.5768e-02,
 5.9232e-02,  7.8471e-02,  9.9465e-02,
 2.4872e-02, -4.4226e-02,  3.2357e-02,
-6.0139e-02, -2.2756e-02, -5.5412e-02,
 4.5363e-02,  1.6393e-01,  3.7428e-02,
 5.2497e-02,  9.5435e-02,  9.7155e-02,
 8.2849e-02,  5.9711e-02,  1.4352e-01,
 1.1756e-02,  1.5440e-02,  1.3039e-01,
 4.3324e-03,  5.9119e-02,  1.1129e-01,
-3.9591e-03,  5.8617e-02, -1.3843e-02,
-2.9949e-02,  3.4877e-02,  5.0679e-03,
 3.7278e-02, -2.5221e-02,  1.2191e-01,
 1.5626e-01,  8.9797e-02, -1.5458e-02,
 1.5607e-01,  1.4561e-02,  1.1720e-01,
-1.6112e-02,  7.7908e-02, -6.1322e-02,
 3.8589e-39,  3.9262e-39,  3.8641e-39,
 3.9450e-39,  3.8805e-39,  3.9383e-39,
 3.8384e-39,  3.8027e-39,  3.7700e-39,
 6.2294e-02, -5.6804e-03, -4.7293e-01,
 1.3161e-01,  3.1187e-01, -1.8013e-01,
 4.9908e-02,  9.8583e-02,  3.8863e-02,
-1.7400e-39,  3.5779e-39,  5.2800e-39,
-1.6845e-39,  4.7140e-39,  2.4244e-39,
-1.3654e-39,  2.4123e-40, -1.5360e-39,
-1.0409e-39,  1.8590e-39, -5.2161e-41,
-8.5110e-40, -1.7210e-39, -4.6624e-39,
 5.0754e-40, -2.6248e-39, -5.4801e-39,
-4.9486e-39,  2.8984e-39,  4.9357e-39,
-1.4077e-39,  3.8778e-39,  5.8202e-39,
-4.1095e-39,  6.8891e-40,  5.6565e-39,
 3.8021e-39, -5.4740e-41,  2.1795e-39,
-2.4185e-39, -5.8101e-39,  1.5651e-39,
-4.9775e-39,  6.0152e-39, -5.2337e-39,
-4.4350e-39, -3.8239e-39,  3.1624e-40,
-4.3665e-39, -3.0919e-39, -4.7675e-39,
-2.3335e-39,  1.8270e-39, -5.5077e-39,
 5.5906e-39,  6.7732e-41,  3.7359e-39,
-5.1412e-40, -2.3239e-39,  5.1937e-39,
-4.4951e-39, -3.4928e-40, -5.0589e-39,
 4.9149e-39,  1.1372e-39,  6.6368e-40,
-1.8870e-40, -5.9117e-40, -1.3973e-39,
-2.3555e-39, -1.0637e-39,  3.1692e-39,
-4.8054e-39,  4.8090e-40,  2.0873e-39,
 3.8301e-39, -3.8642e-39,  4.8187e-39,
-1.6563e-39,  8.9890e-40, -3.5162e-39,
-2.3010e-01, -7.4445e-02, -1.0006e-01,
-2.4543e-01, -8.5750e-02,  1.4859e-01,
-1.3783e-01,  1.2709e-01,  2.5012e-01,
 1.0310e-01, -2.3520e-02, -8.1277e-02,
-2.9267e-02,  1.0686e-01,  4.6287e-02,
-1.2342e-02, -1.7104e-02,  8.4357e-02,
-1.8492e-02, -2.0711e-02, -3.5242e-02,
 7.6163e-02,  6.0853e-02,  9.4248e-02,
 6.2008e-02,  1.1373e-02,  2.6609e-02,
-7.8135e-02,  1.0672e-01, -5.8380e-02,
 7.1618e-02,  2.7966e-04,  1.1835e-01,
 1.1306e-01, -7.8578e-03,  5.1743e-03,
-1.2123e-01,  4.9640e-02,  7.3827e-02,
-1.0377e-01, -3.7377e-02, -3.6536e-02,
 5.7489e-02, -4.6279e-04,  9.0068e-02,
 4.0784e-05, -3.3328e-02,  5.1191e-02,
 9.6538e-02,  7.1779e-02,  1.2121e-01,
 1.1598e-01, -5.9055e-02,  8.2671e-02,
-1.7292e-39, -1.7848e-39, -1.7308e-39,
-3.2817e-39, -1.7274e-39, -3.3601e-39,
-1.7252e-39, -3.4067e-39, -1.7783e-39,
-7.4053e-02, -4.2785e-01, -4.7597e-01,
 4.6309e-01,  7.6018e-02, -3.5885e-01,
 3.0428e-01,  8.7449e-02,  9.7880e-02,
-3.4191e-02,  1.1834e-01, -4.3273e-02,
-6.0782e-01,  9.2387e-01, -1.3972e-01,
 3.0665e-01,  4.7445e-01,  4.8683e-02,
-1.8865e-02,  9.9509e-02, -4.9881e-02,
 2.1640e-02, -2.0941e-01, -1.4779e-01,
 1.7808e-01, -1.2572e-01, -9.6756e-02,
-1.0143e-01,  8.3153e-02, -1.0478e-01,
 1.6201e-01,  2.0740e-01, -1.2653e-01,
 8.1654e-02, -7.6224e-02, -8.9864e-02,
 4.5383e-02, -3.6893e-02, -1.0096e-01,
 2.0389e-01,  2.2557e-01, -1.9685e-01,
-9.5198e-02,  2.2877e-01,  2.1135e-02,
-1.0919e-01, -1.7563e-01, -3.5255e-01,
-1.3447e-01,  3.3709e-01, -1.9043e-01,
-2.1422e-01, -2.8848e-01, -5.3921e-02,
 5.5351e-02, -5.0579e-02, -1.6168e-01,
 2.5282e-01,  1.9715e-01, -2.4035e-01,
-3.0800e-02,  1.9329e-01, -1.0893e-01,
-3.4416e-39, -1.8080e-39, -1.6625e-39,
-1.6612e-39, -1.7397e-39, -1.5953e-39,
 5.3047e-39,  5.4221e-39, -1.1665e-39,
 2.1838e-02, -7.0635e-02,  3.6095e-01,
 5.1096e-01,  6.3838e-01,  5.0716e-01,
 1.1642e-01,  1.8546e-01,  1.5989e-01,
 1.0799e-01,  2.8380e-01,  1.4910e-01,
-2.4305e-01,  2.3084e-01, -9.9982e-02,
-4.6839e-01,  6.0376e-01, -1.2748e-02,
 8.7608e-02,  9.8828e-02,  2.1469e-02,
-3.5384e-03, -1.5689e-01, -1.1411e-01,
 2.0728e-02,  5.6814e-02, -1.1090e-02,
-3.9301e-02, -9.4325e-02, -6.2119e-02,
 1.2842e-01,  9.7466e-02, -2.7502e-02,
 1.6560e-01,  1.5058e-01,  2.2821e-02,
-8.1287e-02, -6.3940e-03,  3.2162e-02,
 9.4116e-02, -6.2567e-02, -1.2704e-01,
 5.4654e-02,  1.4885e-02,  3.8166e-03,
 1.9830e-01, -2.5419e-01, -6.7067e-02,
 3.2303e-01,  1.6037e-01, -3.0200e-02,
 1.3011e-01,  7.5455e-02, -1.2726e-02,
-1.9198e-01, -1.5419e-01, -7.5420e-02,
 1.6070e-01, -6.1031e-02, -2.0179e-01,
-1.5829e-02,  1.9918e-01,  1.0960e-01,
-5.5215e-39, -5.8659e-39, -5.5573e-39,
-6.2394e-39, -6.0172e-39, -6.0159e-39,
-4.0308e-39, -4.1217e-39, -4.1372e-39,
 1.6143e-01,  1.7271e-01,  4.3534e-01,
-2.4312e-01,  4.0146e-01,  4.4693e-01,
 1.5442e-01,  3.9885e-01, -1.4357e-01,
-6.0236e-02, -1.2324e-01,  6.1197e-02,
-2.5842e-02, -1.0266e-02,  1.5670e-03,
 2.9103e-02,  2.9966e-02,  1.1286e-01,
 3.4528e-02,  1.3039e-01,  9.2736e-02,
 3.5193e-02,  5.6583e-02,  5.9465e-02,
 1.2846e-01,  9.3387e-02,  9.2131e-02,
 1.4974e-03,  1.0196e-01,  6.7632e-02,
 8.9809e-02,  5.7568e-02, -6.0621e-02,
-2.7582e-03,  3.1935e-02,  3.1299e-02,
 1.3595e-01,  4.9498e-02,  1.2535e-01,
-3.9396e-02,  4.8859e-02,  4.1389e-02,
 3.7026e-02,  1.3667e-01,  7.5657e-03,
-5.3476e-02,  1.9677e-02,  9.5214e-02,
 1.3136e-02,  7.5560e-02,  6.2428e-03,
-5.2378e-02, -1.8704e-02,  1.0657e-01,
-4.2938e-02, -5.0199e-02,  1.4357e-01,
-5.7002e-02,  1.4158e-01,  4.9442e-02,
-6.8383e-02,  1.1316e-01,  5.2071e-02,
 1.5031e-40,  2.1250e-40,  1.8673e-40,
 1.5681e-40,  1.3104e-40,  1.6173e-40,
 2.1560e-40,  1.8582e-40,  1.7747e-40,
 8.4848e-02, -1.9845e-01, -5.1844e-01,
 3.0959e-01,  3.6682e-01,  3.1208e-02,
 1.9871e-01,  2.8318e-01,  1.6066e-01
}
,)"
R"(
{
-2.7283e-39, -4.9031e-39, -2.1039e-39,
-1.0327e-39, -5.1679e-39, -4.3300e-39,
-5.2613e-39, -3.1707e-39, -6.0916e-39,
 1.5840e-39,  1.6709e-39,  1.6120e-39,
 1.6716e-39,  1.7418e-39,  1.6624e-39,
 1.5922e-39,  1.7383e-39,  1.5668e-39,
 1.1389e-01, -4.5774e-02,  6.1423e-02,
 1.3858e-01,  2.3102e-02, -6.5079e-02,
 1.3269e-01,  3.2387e-02,  7.6966e-02,
-2.1531e-39, -1.6063e-39, -3.2070e-39,
-2.8531e-39,  4.6956e-39,  1.4038e-39,
 2.0509e-39, -4.4924e-39, -5.3658e-39,
 1.1524e-01, -5.0115e-02,  9.4187e-02,
 4.2477e-02,  1.4197e-01,  2.4986e-02,
-2.8688e-02,  9.2289e-02,  4.1965e-02,
-2.1691e-01, -6.6916e-04, -1.3026e-01,
-1.9143e-01,  1.2211e-01,  1.2562e-01,
-1.2273e-01,  7.1045e-02,  1.2396e-01,
-8.0861e-02, -4.4301e-03,  6.3144e-03,
 3.0338e-02, -8.6463e-03,  5.5084e-02,
-1.8370e-01, -5.0287e-02, -7.2194e-02,
 7.4570e-02,  5.4483e-02, -1.2639e-02,
 1.2481e-01,  1.4683e-01, -4.7581e-02,
 1.6748e-01, -3.1374e-02, -1.7271e-02,
 1.9801e-39, -3.3469e-39, -4.7012e-39,
-2.9869e-39, -3.2752e-39, -2.2142e-39,
-4.2927e-39, -1.9635e-39, -8.7517e-40,
 2.7286e-39,  2.7755e-39,  2.7501e-39,
 2.7114e-39,  2.7711e-39,  2.6858e-39,
 2.5562e-39,  2.6523e-39,  2.5846e-39,
 1.4015e-01,  1.0486e-01,  1.2320e-01,
 4.6545e-02,  1.2068e-01,  9.2531e-02,
 1.0717e-01,  3.8738e-02,  1.0181e-01,
-7.4503e-40, -1.1490e-39,  6.1230e-41,
 2.4896e-39,  5.3740e-39, -1.4060e-39,
 1.9095e-39, -7.1020e-40,  3.5820e-39,
-1.4348e-02,  6.4128e-02,  6.1082e-02,
-1.1112e-02,  8.5993e-02,  2.4835e-02,
 1.2794e-01, -9.1072e-02, -1.3487e-02,
-5.8057e-02,  1.3080e-01,  1.0895e-01,
-1.6436e-01,  9.8593e-03,  1.5586e-02,
-1.5336e-01,  3.6391e-02,  1.4539e-01,
-4.6112e-02,  3.0102e-02,  6.2460e-02,
-2.5510e-02,  2.0437e-02, -5.6816e-02,
-1.0308e-01, -1.5284e-01, -7.1036e-02,
 5.5290e-02, -6.6632e-02,  4.2268e-02,
-2.7665e-02,  9.3415e-02,  5.1026e-02,
 1.5652e-01,  1.0835e-01,  9.6131e-02,
-4.2583e-39, -3.4889e-39, -5.7522e-39,
 4.2701e-40,  2.8095e-39, -3.5579e-39,
 2.2286e-39,  4.9865e-39,  4.0469e-39,
-6.4320e-40, -3.3384e-39, -5.9025e-39,
-7.9075e-40, -3.0577e-39, -6.0007e-39,
-8.9627e-40, -2.8374e-39, -5.8866e-39,
 6.3645e-03, -5.3080e-03, -5.1759e-02,
 1.0665e-01, -6.3126e-02,  5.0918e-02,
 7.2193e-02, -6.8836e-02, -6.5657e-02,
 2.8519e-39, -5.0955e-39, -9.6085e-40,
-3.3563e-39, -5.6038e-39, -1.6256e-39,
 2.6872e-39,  1.4728e-39, -1.9908e-39,
-1.5254e-02,  9.8323e-02,  4.5504e-02,
 1.3855e-01,  6.9300e-02,  1.9135e-01,
-5.2321e-02, -6.0227e-03, -1.1734e-04,
-1.4457e-01,  9.2761e-02,  4.5219e-02,
-3.0361e-01,  3.4673e-01, -2.3110e-01,
 2.1017e-01,  2.4983e-01,  3.1659e-01,
-6.0569e-02, -5.4348e-02, -7.6719e-02,
-6.5060e-02,  2.8902e-01,  8.0732e-02,
-3.3425e-01, -3.1361e-01, -2.7183e-01,
 2.8035e-02, -5.8134e-02, -4.3880e-02,
-1.6375e-02,  9.8195e-02, -7.4011e-02,
-5.9523e-02,  1.0234e-01, -5.3357e-02,
 2.3364e-39, -2.5324e-39, -4.8333e-40,
 2.2903e-41, -3.3061e-39, -2.5779e-39,
-1.8164e-39, -4.9236e-39, -4.9272e-39,
-1.2809e-39, -1.1698e-39, -1.2564e-39,
-1.3111e-39, -1.1778e-39, -1.2543e-39,
-1.4772e-39, -1.4021e-39, -1.4721e-39,
 8.8919e-02, -3.4541e-03, -4.9619e-02,
 1.0997e-01,  1.0257e-01,  6.9950e-02,
 9.2624e-02,  3.2712e-02,  8.7916e-02,
-5.0242e-39, -6.1320e-39,  8.7891e-40,
-4.9951e-39,  2.3873e-39, -2.7823e-39,
-3.6739e-39, -1.8903e-39,  5.2150e-39,
 9.6288e-02,  9.7568e-03, -5.8178e-02,
 2.3313e-02,  1.1725e-01,  1.0291e-01,
-1.0111e-01,  8.3706e-02,  9.6575e-03,
-8.2531e-02,  7.0089e-02,  1.0821e-01,
-1.1016e-01,  1.8977e-01,  2.5576e-01,
-1.0221e-01,  5.9236e-02,  6.1678e-02,
 2.6234e-02,  9.6868e-02,  9.2432e-02,
 4.9881e-02,  5.9121e-02, -1.0477e-02,
-1.4693e-01, -1.0030e-01, -1.0608e-01,
 1.1936e-01, -2.2301e-02,  1.1363e-01,
 1.3981e-01,  6.7734e-02, -8.2775e-02,
 1.0404e-01, -7.7360e-03,  4.2523e-02,
-2.6052e-39,  5.7201e-39, -5.6049e-39,
-3.6314e-39, -5.9232e-39, -3.6970e-39,
 3.4360e-39, -5.6848e-39, -3.8308e-39,
 4.6279e-39,  5.8135e-39,  2.0652e-39,
 3.9864e-39,  4.4000e-39,  5.5163e-39,
 2.9644e-39,  2.7537e-39,  3.6593e-39,
 4.7872e-02, -2.5857e-02,  4.8810e-02,
 1.0389e-01, -1.0782e-01,  4.1365e-02,
 9.5778e-02, -5.2341e-02,  4.5947e-02,
-8.2652e-40, -5.7602e-39,  4.6187e-39,
-2.8365e-39,  1.4981e-39,  6.2504e-39,
-4.8330e-39,  4.0283e-39,  4.9792e-39,
-1.0893e-03, -8.2708e-02, -1.7925e-01,
 8.3461e-02,  3.1339e-02,  8.8096e-02,
 7.3139e-02, -1.2212e-01,  1.0489e-02,
-2.4187e-01, -3.8397e-01,  1.3730e-01,
 1.9217e-01,  1.4101e-01,  4.9795e-01,
-1.1441e-01,  3.3343e-01,  7.9194e-02,
 1.4556e-01, -5.1060e-01,  2.1556e-01,
 3.5719e-01,  2.7282e-01, -1.9015e-01,
-1.0941e-01,  2.7634e-02,  1.1833e-01,
-9.3316e-02, -4.1307e-03,  7.8613e-02,
-2.1526e-02, -6.7141e-02,  2.5513e-02,
-3.3942e-02, -8.6282e-02,  3.0446e-02,
-4.5124e-39, -2.7154e-39,  4.9467e-39,
-4.2299e-39, -5.9485e-39, -2.9606e-39,
-4.7642e-39, -4.7981e-39, -4.0169e-39,
-3.8238e-39,  5.7381e-39,  4.0097e-39,
 1.9550e-39,  4.5523e-39,  3.1206e-39,
 6.0200e-39,  3.0406e-39,  2.0498e-39,
-3.2474e-01,  1.1052e-02,  4.7197e-02,
-1.4658e-01,  1.6728e-01,  5.2190e-02,
 4.3174e-02,  4.5864e-02,  5.4472e-02,
 2.6403e-39,  2.7421e-39, -4.3011e-39,
-3.6258e-39, -1.3708e-39,  3.6147e-39,
-1.9471e-39,  4.5896e-39,  4.5992e-39,
-9.9986e-02,  7.0727e-02,  8.5023e-02,
 2.2501e-02,  1.4343e-01,  1.1878e-01,
 2.8126e-02,  7.3239e-02,  1.0468e-02,
 4.5032e-01,  4.4730e-01,  1.3446e-01,
-1.3374e-01,  8.8554e-02,  3.5610e-01,
 3.0584e-01,  2.3536e-01,  1.6161e-01,
-5.1485e-01,  1.2372e-01,  5.4379e-02,
-2.9665e-01, -3.3157e-02, -1.8688e-01,
 5.1777e-02, -1.4315e-01, -1.1366e-01,
-2.4471e-01,  5.5554e-02,  8.9284e-02,
-1.6870e-01,  7.6156e-02,  1.2472e-01,
-1.5633e-01,  4.3184e-03,  1.1078e-01,
 4.0579e-39, -3.8271e-39,  1.1535e-39,
 6.6968e-40, -1.1545e-39, -5.4217e-40,
 3.5566e-39, -4.4956e-40, -1.7097e-39,
-4.1778e-39, -3.7655e-39, -3.7148e-39,
-3.8013e-39, -3.5225e-39, -3.4678e-39,
-3.8369e-39, -3.5583e-39, -3.6518e-39,
-1.4894e-02,  2.4801e-03, -4.6996e-02,
 6.7453e-04,  1.8799e-02,  2.9889e-02,
 7.2700e-03,  1.2385e-01,  9.2522e-02,
 3.9300e-39,  3.1853e-39,  2.8376e-39,
 2.8888e-39, -4.8734e-39,  2.3402e-39,
-3.9710e-39, -4.3243e-39,  4.1151e-39,
 1.6399e-02, -8.2828e-02, -5.8361e-02,
 2.1315e-02,  1.1968e-02,  6.8727e-02,
 3.8558e-02,  1.5451e-02,  5.4465e-04,
 1.0549e-02, -8.6468e-02, -1.8535e-01,
-1.3616e-01,  2.7371e-01,  1.1157e-01,
-1.7097e-01,  1.3659e-01,  2.2831e-02,
-3.3897e-02,  1.3307e-01,  7.4482e-03,
 4.8120e-01,  7.7053e-01,  5.3354e-01,
-2.4277e-01, -5.9136e-02, -1.3419e-01,
-7.4653e-02, -6.4169e-02, -2.9526e-02,
-3.6336e-02,  7.2362e-02, -3.5332e-02,
 6.2628e-02,  6.2278e-02,  3.5639e-02,
 3.6614e-39, -2.6150e-39, -3.5229e-39,
 5.3538e-39, -1.2368e-39,  2.1530e-39,
 4.8585e-39, -2.4150e-39,  5.2220e-40,
 3.8610e-40,  1.4772e-39,  2.1962e-39,
-1.8493e-40,  1.1409e-39,  1.7309e-39,
-2.5751e-40,  9.1351e-40,  1.3106e-39,
 6.2867e-02, -1.2727e-01, -6.5307e-02,
 1.1415e-01, -4.5529e-02, -1.1358e-01,
 4.3427e-02, -6.0994e-02, -7.7808e-02,
-4.1831e-39,  1.3230e-39,  5.5853e-39,
-3.4646e-39, -7.2824e-40, -3.4263e-39,
 1.5344e-39, -5.8245e-39,  1.9910e-39,
 1.1000e-02, -3.7088e-03, -8.0042e-02,
 9.7603e-02,  8.6581e-02, -1.8921e-03,
 2.2820e-01,  6.8073e-02, -8.1081e-02,
-3.3901e-01, -1.1231e-01, -8.6476e-02,
 1.1147e-01,  4.9587e-01, -1.7039e-01,
-2.0702e-01,  5.8730e-02, -1.3475e-01,
 2.3548e-01, -6.8044e-02,  9.4296e-02,
 4.4803e-01,  6.1517e-03, -5.5192e-02,
-2.7304e-01, -2.6003e-02,  4.0713e-01,
 2.8621e-02,  6.2698e-03, -1.4746e-01,
 9.4819e-02, -1.3109e-02,  3.5540e-02,
 4.4047e-02,  3.5066e-02, -9.5886e-03
}
,)"
R"(
{
-6.7011e-03,  1.7398e-01,  1.4767e-01,
-1.9882e-02,  1.9286e-01,  4.8626e-02,
 1.1465e-01, -4.4017e-02, -1.9288e-01,
-7.5817e-02,  1.5598e-01,  1.2329e-01,
 3.4126e-03, -9.4884e-02, -4.2276e-02,
 3.9110e-02, -1.3477e-01, -4.4951e-02,
 6.0450e-02,  4.4656e-01,  3.8954e-01,
-2.1207e-01, -1.0600e-02, -5.6351e-01,
 1.8074e-01,  3.0797e-02, -4.0380e-01,
-1.0733e-01,  3.7228e-02,  9.7157e-02,
-7.5810e-03,  5.5605e-02, -9.1898e-02,
-1.4992e-01, -5.3206e-02, -1.9667e-01,
-1.6667e-01,  7.6091e-02,  1.7064e-01,
 2.5322e-01, -9.4636e-03, -2.7899e-01,
 4.2013e-02,  1.5693e-01,  3.1124e-01,
-2.1534e-02,  1.3915e-01, -2.8199e-01,
-2.9683e-03,  1.4445e-02, -1.5552e-01,
 3.4759e-02, -2.0321e-01, -1.1155e-01,
 3.6164e-02,  2.8664e-01,  2.3426e-01,
-1.2525e-01, -1.7195e-01, -5.2270e-02,
 3.8782e-02,  5.7734e-02,  2.1945e-01,
 1.0243e-01, -1.3159e-01, -1.7844e-01,
-6.0359e-02,  1.9125e-01,  3.3553e-01,
-1.0876e-01, -1.2149e-01, -5.7185e-01,
-2.0583e-02, -4.8168e-03, -7.1908e-02,
-2.3428e-02,  2.9902e-02,  1.0888e-02,
 3.6383e-02,  1.0052e-01,  2.8972e-02,
 1.1415e-03, -3.4518e-02, -9.0058e-02,
 7.3207e-03,  6.0961e-02,  7.5629e-02,
-4.5969e-02,  2.4314e-02,  6.7658e-02,
-1.3043e-01, -3.0343e-01, -2.0799e-01,
-4.6261e-02, -1.7650e-02, -7.2160e-02,
-2.6291e-02,  1.5707e-01,  9.5021e-02,
-4.1030e-02, -8.1977e-02, -3.0776e-02,
-3.0685e-02,  8.2163e-03,  4.0357e-02,
-6.9633e-02,  6.0690e-02,  1.5418e-02,
-1.2814e-01,  7.3968e-02, -3.3742e-03,
-1.5239e-01,  8.9941e-03,  1.7877e-01,
 2.1219e-01, -5.2057e-01, -2.2284e-01,
-3.4681e-02, -1.3594e-02,  1.6700e-01,
-7.7366e-02,  8.5138e-03, -4.3159e-02,
 4.0597e-02,  9.7247e-04, -3.4326e-01,
-2.1424e-01, -1.6489e-01, -4.3248e-02,
 1.5987e-01,  4.6235e-01,  2.6287e-01,
-1.2270e-02,  1.3165e-01,  5.3217e-02,
 7.2716e-02, -7.0677e-02, -1.7740e-01,
-6.2357e-02,  1.1932e-01,  1.5733e-01,
-1.0275e-01,  1.4966e-01,  4.8125e-02,
-4.7150e-02,  1.5516e-01,  6.9615e-02,
 6.1252e-02,  5.3859e-02,  1.7052e-01,
 3.1940e-02,  1.1842e-01,  4.2265e-02,
-4.9531e-02,  1.1519e-01,  9.8914e-02,
 1.3455e-01,  1.3177e-01, -2.7938e-03,
 1.1895e-01,  1.1377e-01,  6.1035e-02,
 8.0390e-02, -4.1028e-02,  3.7415e-03,
-1.0317e-01,  1.0279e-01, -6.5789e-03,
-2.3339e-02,  7.2741e-02,  4.1662e-02,
-7.4087e-02,  8.8531e-02, -4.9697e-02,
 4.6134e-02,  1.4300e-01,  1.1720e-01,
 3.8271e-03,  1.7108e-01, -2.4779e-02,
 6.9844e-02, -4.6467e-02, -9.1699e-02,
 5.5704e-02, -3.0312e-02, -7.8252e-03,
-4.3799e-02, -1.6623e-01, -2.3006e-02,
 4.9214e-02,  3.1528e-02,  3.3302e-02,
 3.1213e-02,  9.8880e-02, -1.1098e-01,
 4.5092e-02, -1.6922e-03, -5.1380e-02,
 7.6063e-02,  1.4159e-01,  4.1409e-02,
 8.0812e-02,  9.7569e-02,  4.1532e-02,
-1.1136e-01, -4.3686e-02, -1.4144e-01,
-9.7717e-02,  4.8239e-02,  5.3374e-02,
-1.1827e-01,  1.0008e-01,  8.6368e-02,
-6.2572e-02,  3.6484e-02, -6.3361e-02,
 4.1008e-03,  1.6709e-02,  4.0553e-02,
 2.2766e-02,  2.7241e-02,  5.1786e-02,
 1.3607e-02,  5.4638e-02,  6.9439e-02,
-2.4211e-02,  4.0065e-03, -1.9540e-03,
-9.5697e-03,  3.0503e-02,  3.5809e-02,
-4.3456e-02,  2.8959e-02,  4.2898e-02,
-1.5629e-02, -9.4347e-02,  7.2799e-02,
 2.3115e-01,  7.3449e-02,  6.9354e-02,
 1.6014e-01,  1.8878e-01, -2.2148e-02,
-4.9274e-02, -6.9233e-03,  1.0578e-02,
-4.3291e-02, -7.8361e-03,  1.6647e-02,
-5.6168e-02,  1.0317e-02,  3.1170e-02,
 1.2530e-01, -3.2398e-02, -6.5690e-02,
-2.5805e-01,  3.6079e-02,  3.5390e-02,
-1.7236e-01,  6.6798e-03,  4.8924e-02,
 1.3314e-01,  5.0646e-02, -3.4844e-02,
-1.2559e-01, -1.1774e-01,  1.2898e-01,
-7.7402e-02, -1.0703e-02, -2.6359e-01,
-3.8706e-02, -2.2082e-02,  2.7591e-03,
-8.2353e-02, -3.1941e-02, -1.1937e-01,
 2.9747e-02,  2.0041e-01, -5.1984e-02,
 1.7919e-01,  6.3603e-02, -5.5516e-02,
 1.0116e-01,  8.7370e-02, -8.6624e-02,
-8.4314e-02,  3.5997e-02,  2.1161e-01,
 1.0902e-39,  9.3514e-40,  9.3074e-40,
 9.8377e-40,  1.1299e-39,  8.2024e-40,
 1.2062e-39,  1.0405e-39,  1.0284e-39,
-5.7829e-40, -6.7489e-40, -6.3814e-40,
-6.8460e-40, -7.9377e-40, -7.6449e-40,
-4.7632e-40, -5.6022e-40, -5.2053e-40,
 1.8459e-39,  2.1036e-39,  2.1848e-39,
 2.0535e-39,  2.3728e-39,  2.4416e-39,
 1.7027e-39,  2.0249e-39,  2.0833e-39,
 9.1594e-40,  8.0493e-40,  7.7836e-40,
 7.5889e-40,  6.3026e-40,  9.3384e-40,
 9.6987e-40,  1.1273e-39,  8.1906e-40,
-7.9046e-39, -7.2328e-39, -7.1040e-39,
-7.9046e-39, -7.1862e-39, -7.4931e-39,
-6.5243e-39, -7.1117e-39, -6.9941e-39,
 1.3577e-39,  3.5945e-40, -3.6833e-40,
 1.3768e-39,  6.9779e-40, -7.5180e-40,
 5.7295e-40, -6.0767e-41, -1.3085e-39,
 7.7960e-39,  7.8579e-39,  7.4482e-39,
 7.4224e-39,  7.5791e-39,  7.4378e-39,
 6.5819e-39,  6.7271e-39,  6.6281e-39,
-1.6535e-39, -7.7817e-40, -8.5918e-40,
-2.0861e-39, -1.3658e-39, -1.0560e-39,
-3.4360e-39, -2.6878e-39, -2.6477e-39,
 4.6460e-02,  1.1676e-01, -5.9846e-02,
 8.6467e-03, -1.1287e-02,  7.0129e-02,
-1.1277e-01,  1.0321e-02, -1.9567e-02,
 1.2145e-01, -7.1995e-02, -1.3615e-02,
 9.7877e-02,  6.6061e-02,  1.0272e-02,
 1.1391e-01,  5.6974e-02,  9.7472e-02,
-3.3605e-02,  6.1751e-02, -4.3004e-02,
-5.1040e-02, -3.8798e-02, -7.1736e-02,
-1.0179e-02,  8.5964e-02, -8.1435e-04,
 2.5149e-02,  7.1990e-02,  8.1534e-02,
 6.3133e-02,  5.8643e-02,  4.6756e-02,
-5.3580e-03,  3.4411e-02,  5.2957e-03,
 1.0652e-01, -6.6035e-02,  8.5754e-02,
 3.2919e-01, -1.5958e-02,  2.1694e-03,
-9.0943e-02, -2.1920e-02,  2.9706e-02,
 4.7986e-02,  1.7105e-02, -5.7711e-02,
-4.2066e-03,  6.5668e-02, -1.6617e-01,
 1.0057e-02, -2.0108e-03, -1.5499e-01,
 6.7941e-02,  1.7352e-01,  4.9498e-02,
 6.2013e-02,  9.6180e-02, -2.9861e-03,
-1.2482e-02,  9.5709e-03, -8.7913e-02,
-8.6954e-02,  9.9646e-03,  8.0050e-02,
-4.4157e-02, -6.3008e-03,  4.0645e-02,
-7.9624e-02,  1.0856e-01, -4.5341e-04,
 7.1085e-02,  5.7002e-02,  1.1673e-02,
-5.1378e-02, -2.3945e-03, -5.9532e-02,
 3.4998e-02, -3.6019e-02,  1.0428e-02,
 5.9774e-03,  5.4993e-03,  2.4306e-02,
-5.9813e-03,  4.4999e-02,  7.4744e-02,
-3.0773e-02, -3.6835e-02,  5.8396e-04,
-3.8644e-01,  2.4563e-01,  1.2436e-01,
-3.2986e-01, -1.1044e-01,  2.0753e-01,
-1.3621e-01, -1.3544e-01,  5.8882e-02,
 8.8837e-02,  5.7460e-02, -3.0960e-02,
-1.2598e-03,  3.9124e-02, -5.3322e-02,
-4.4227e-02, -3.8000e-02, -3.2677e-02,
 1.5675e-01,  1.0808e-01,  1.1024e-01,
 5.4468e-01, -5.9268e-01,  1.0088e-01,
 8.2360e-02,  1.9646e-01,  6.4799e-03,
 1.6357e-01,  6.8273e-02, -1.2051e-01,
 4.9511e-02,  4.7334e-01, -4.8876e-02,
-1.3130e-01, -5.1568e-03,  1.0088e-01,
-5.8971e-02,  2.5775e-01,  9.0169e-02,
-3.0461e-01, -3.2353e-02, -2.0293e-01,
 1.3897e-02,  1.4249e-01, -5.8661e-02,
-1.3624e-01, -5.3026e-02,  3.1038e-03,
-5.6211e-01, -2.8375e-01, -1.2524e-01,
-2.3813e-01, -2.2439e-02, -4.4082e-02,
 9.9066e-02, -7.1735e-02,  2.2345e-02,
-1.4791e-02,  1.3225e-01,  8.9460e-02,
-4.8986e-02, -3.2296e-02, -4.7474e-02,
 6.5865e-02, -8.0697e-02, -6.8475e-02,
-7.6845e-02,  1.1568e-01,  3.7443e-03,
 1.0448e-01, -3.3206e-03,  5.4523e-02,
 5.5741e-02,  5.0917e-02,  1.0209e-01,
-9.6729e-02,  7.8876e-02, -4.9550e-02,
-3.8926e-02,  7.1163e-02,  8.9436e-02,
-1.4001e-03, -9.4980e-02, -7.7747e-02,
 9.4335e-02,  1.1605e-01,  9.5715e-02,
 1.7951e-02,  4.3177e-03, -5.6937e-02,
 4.4558e-02, -5.2562e-02,  4.0652e-02,
 1.8058e-01, -1.0763e-01,  4.8927e-02,
-5.2569e-03, -1.3437e-01,  2.8578e-02,
 1.3592e-02, -3.9346e-02,  1.0003e-01,
 1.8091e-01,  7.2687e-03, -3.7241e-02,
 6.0438e-02,  5.7872e-02,  7.3778e-02,
 1.2411e-02,  4.1856e-02, -2.8892e-02,
 3.2884e-02,  6.9072e-02, -5.9363e-02,
-1.7112e-01, -9.9734e-02, -7.3417e-02,
-8.9623e-02,  4.5292e-02, -1.6635e-01,
-3.1895e-02,  1.4284e-01,  2.0752e-01,
 2.3383e-02, -1.3490e-02,  5.1593e-03
}
,)"
R"(
{
 5.8708e-01,  2.6026e-01,  8.8379e-02,
 3.1818e-01,  7.0055e-03,  1.1652e-01,
 1.1719e-01,  8.7711e-02, -1.1687e-02,
 7.5741e-02, -3.7970e-01,  1.6001e-01,
 1.0739e-01,  3.1735e-01,  2.0061e-01,
 8.6719e-02,  8.5111e-02, -3.9354e-02,
-9.9512e-02, -9.1524e-02, -9.7984e-02,
 5.6333e-02, -1.5928e-01,  1.1998e-03,
 2.7488e-02,  2.8168e-02,  1.3768e-01,
 5.9686e-02,  2.8931e-01, -1.7131e-02,
 1.6391e-01,  3.3748e-01,  1.2296e-01,
 8.9242e-02,  1.4761e-01,  1.7187e-01,
-2.6352e-39, -4.0703e-39, -5.1751e-39,
-2.5214e-39, -3.9666e-39, -4.6282e-39,
-2.4635e-39, -3.6734e-39, -4.3359e-39,
-7.1654e-02,  7.9691e-03, -1.0219e-01,
-5.5684e-02, -1.3065e-01, -1.9106e-02,
 1.0561e-01,  5.9054e-02, -2.1279e-02,
-1.8840e-02,  1.6690e-01,  3.8050e-01,
 6.2779e-02, -1.2124e-01,  5.0304e-01,
 2.1870e-02,  1.7631e-01,  1.4858e-01,
 1.4614e-01, -1.1767e-01, -3.9155e-02,
 1.2963e-01, -4.6753e-02,  1.3848e-01,
-8.2292e-02,  2.1908e-01,  6.2794e-02,
-3.2625e-01, -8.8528e-03, -6.5603e-03,
 5.4245e-02,  2.7983e-01,  2.1608e-01,
 8.5890e-02,  1.0955e-01, -1.1606e-01,
 9.7435e-02,  1.5911e-01,  6.7285e-02,
 3.9570e-02,  1.9333e-01, -1.5531e-02,
-2.3475e-01, -2.5006e-02,  2.8106e-02,
 6.8740e-03,  1.3261e-01, -3.8563e-02,
 8.8758e-02, -4.2225e-02,  4.7042e-02,
 5.6284e-02, -2.8303e-02,  3.4532e-03,
-4.0265e-02, -3.0645e-02, -5.2059e-02,
-4.6196e-02, -2.4868e-02, -3.3257e-02,
-3.7208e-02, -2.4100e-03, -7.1959e-04,
 6.4237e-39,  6.1438e-39,  6.5434e-39,
 6.1596e-39,  6.1608e-39,  6.3157e-39,
 6.4263e-39,  6.4625e-39,  6.5877e-39,
 1.1092e-01, -4.4784e-02,  9.1292e-02,
 9.2900e-02,  1.2459e-01, -7.1447e-02,
 2.6158e-02, -5.0219e-02, -5.6136e-02,
-5.8603e-02,  2.9323e-02, -2.4230e-01,
-9.4921e-02,  1.9103e-01,  1.1670e-01,
 1.2022e-02,  6.2830e-02,  3.0393e-01,
 3.3819e-02,  1.0040e-01,  8.2600e-02,
-8.7604e-02,  7.0641e-02, -1.0132e-01,
-9.9371e-02,  8.9363e-02, -1.0703e-01,
 4.4603e-01,  7.9636e-03,  1.8834e-01,
 1.1859e-01,  4.0760e-01,  9.6841e-02,
-1.1735e-01,  2.3993e-01, -7.7916e-02,
 6.3481e-02, -1.4958e-01,  1.1554e-02,
 5.2668e-02,  3.4379e-01,  8.3536e-03,
-5.5403e-02,  1.1655e-01, -7.5022e-02,
-8.2992e-02, -7.0322e-02, -1.0078e-01,
-1.4516e-02, -1.6558e-02,  6.6806e-02,
-6.7454e-04, -5.7525e-02,  1.5772e-01,
 1.6446e-01, -1.1897e-02, -8.3387e-02,
 7.1339e-02,  1.6254e-01,  1.6963e-01,
 1.2630e-02,  5.7933e-02,  8.4686e-02,
-5.6318e-39, -6.1837e-39, -6.1661e-39,
-5.9923e-39, -6.2371e-39, -6.4922e-39,
-6.4206e-39, -6.6092e-39, -7.1603e-39,
 4.6507e-02, -4.5924e-02, -7.3838e-02,
-3.3012e-02,  5.1295e-02, -7.4884e-02,
 7.5389e-02,  1.2002e-01,  3.9442e-03,
 9.9461e-02,  1.9607e-01,  1.4896e-01,
-1.1191e-02,  1.8352e-01,  2.6778e-01,
 8.0977e-02,  1.0885e-01,  2.5331e-01,
 3.1503e-02, -3.0004e-01, -6.9114e-02,
 2.0705e-01, -2.0978e-02,  1.5154e-01,
 6.3033e-02, -1.5721e-01,  5.1067e-02,
-1.1220e-02,  1.5315e-01,  4.5277e-03,
 3.3250e-01,  1.4207e-01,  1.3469e-01,
 5.2996e-01, -2.5803e-01, -4.5525e-02,
 3.9807e-02, -1.7088e-01, -1.2414e-01,
 2.1564e-01, -2.9160e-01, -1.8796e-01,
 1.5482e-02,  2.7005e-01,  8.2446e-02,
 5.4906e-02, -1.0507e-01, -8.0069e-02,
-4.5729e-03, -2.0621e-02,  5.0088e-02,
 2.5479e-02,  9.5924e-02,  8.3813e-02,
 4.7833e-02, -2.6191e-01,  3.3483e-02,
 6.1653e-02,  7.1940e-03, -1.3578e-01,
 1.7662e-01, -2.8194e-02, -2.7509e-02,
-1.9419e-39, -2.4904e-39, -2.7567e-39,
-2.9896e-39, -3.2700e-39, -3.6336e-39,
-3.8942e-39, -4.2028e-39, -4.5229e-39,
-1.6839e-02, -9.4421e-02, -3.0147e-02,
-6.5974e-02, -1.6716e-02,  5.0672e-02,
-7.9841e-02, -4.7086e-03,  5.0016e-02,
 1.8223e-04,  3.3984e-03,  5.1965e-02,
-7.3512e-02, -5.6604e-03, -1.1630e-01,
-1.0767e-01,  3.2261e-02, -2.0044e-01,
 1.0995e-01,  4.3581e-02, -3.9397e-02,
-1.4476e-02, -2.3087e-02,  2.6423e-03,
 1.2047e-02,  1.2084e-01,  1.8563e-01,
-2.8497e-01, -2.5353e-01,  1.0933e-01,
 8.8974e-03,  1.3315e-01,  1.9153e-01,
 2.0427e-02, -8.9900e-02,  2.2363e-02,
 2.8575e-02,  1.6351e-01,  1.1876e-01,
-2.7438e-02, -1.0816e-03, -5.5680e-02,
 5.1369e-02, -2.0575e-02,  4.5232e-02,
 9.4988e-02,  2.5418e-02,  8.9888e-02,
 9.6631e-02,  1.5828e-01,  1.1577e-01,
-2.9665e-02,  3.2035e-02,  1.4428e-01,
 7.4352e-03,  2.4917e-03,  4.2713e-03,
 1.2534e-02,  2.1314e-02,  1.5963e-02,
 2.2920e-03,  2.1864e-02,  2.2921e-02,
 7.1089e-40,  5.3581e-40,  4.5922e-40,
 6.2492e-40,  4.6365e-40,  4.5466e-40,
 9.2740e-40,  7.7219e-40,  7.4187e-40,
-7.0909e-02,  1.1127e-01, -8.8953e-02,
-5.0537e-04,  4.5664e-05,  1.3829e-02,
 7.4380e-02,  1.3900e-03,  4.0345e-02,
 5.7173e-02,  8.7514e-02, -3.9945e-01,
 4.4116e-02,  1.4148e-01, -2.7578e-02,
-1.2133e-02,  1.9647e-01, -2.6767e-02,
 8.5870e-02, -1.3723e-02,  1.3408e-02,
 7.9471e-03,  7.8321e-02,  5.1118e-02,
-8.3660e-02, -7.1584e-02,  2.7423e-02,
-5.5651e-39, -3.2350e-39,  4.7534e-39,
-4.8581e-39, -5.8010e-39,  6.3268e-39,
-3.4016e-39,  6.2313e-39,  5.7413e-39,
-3.0708e-39,  6.0155e-39, -6.3317e-39,
-3.1054e-39, -5.5914e-39, -6.4181e-39,
-1.3636e-40, -6.0343e-39, -6.2034e-39,
 1.0108e-39, -2.5283e-39, -8.6098e-40,
 1.0088e-39, -2.3042e-39, -8.2029e-40,
 1.2802e-39, -3.7761e-39, -4.6451e-40,
 1.4160e-39,  7.3869e-40,  1.3275e-39,
 1.2560e-39,  1.0078e-39,  1.2296e-39,
-2.4490e-39,  8.6071e-40, -2.4510e-39,
 2.1753e-39, -2.0576e-39, -2.1365e-39,
 2.0157e-39,  2.0755e-39,  1.9439e-39,
 2.0998e-39,  2.0732e-39,  2.1072e-39,
-1.1289e-39, -1.6132e-39,  4.8117e-40,
 1.2029e-39, -1.3112e-39,  6.4761e-40,
 1.4958e-39, -9.2719e-40,  8.9526e-40,
 3.6032e-39, -4.9803e-39, -2.4410e-39,
-1.6429e-39, -4.9602e-39, -5.9626e-39,
-1.6627e-39, -4.9809e-39, -5.6258e-39,
 1.6619e-39,  1.7856e-39,  5.1822e-39,
 1.5443e-39,  1.4215e-39,  6.1830e-39,
 1.4242e-39, -1.7895e-39,  5.2206e-39,
-2.4764e-01, -2.8696e-01, -5.7562e-03,
 1.9255e-01,  5.1335e-02, -1.4512e-01,
-1.1017e-02, -3.6505e-02, -1.1773e-01,
 5.8651e-02, -1.9354e-02,  2.1595e-02,
-3.5114e-03,  1.8335e-01,  4.0043e-02,
 1.0579e-01, -6.3055e-02,  2.6981e-02,
-1.4351e-02, -1.5029e-02, -9.7792e-02,
 4.6718e-02,  3.8673e-02, -2.3410e-02,
-2.8942e-03, -8.4898e-03, -3.3613e-02,
 2.0298e-01,  9.7218e-02,  1.5052e-01,
 3.2108e-01,  2.6568e-01,  1.3809e-03,
 1.0008e-01,  6.9262e-02, -4.7810e-02,
 4.1291e-39,  4.3762e-39,  4.2724e-39,
 4.5864e-39,  4.7827e-39,  4.8821e-39,
 4.5529e-39,  4.6921e-39,  4.7519e-39,
 9.1246e-03, -1.8136e-02, -5.8517e-03,
 9.1080e-03,  4.2591e-02, -1.5604e-02,
-3.6270e-02,  5.9184e-02,  2.3189e-02,
 4.2636e-02,  3.6600e-01,  4.7134e-01,
 3.6666e-02,  4.3565e-01,  2.1105e-01,
-5.2747e-02,  4.0503e-01,  2.0926e-01,
 8.8427e-02,  4.9138e-02, -2.3381e-01,
-5.6521e-02,  7.5013e-02, -1.4783e-01,
-4.7299e-02, -8.1200e-02, -6.5665e-02,
-1.6281e-01, -2.3070e-01,  5.4033e-02,
 1.1527e-01,  3.4730e-01,  1.9293e-02,
-1.8352e-02,  2.0626e-01, -1.1955e-01,
 8.1665e-02,  3.8584e-02,  2.7958e-03,
 6.4294e-02,  1.3912e-01, -5.6370e-02,
-1.7618e-02,  9.0357e-02, -5.5021e-03,
 9.3211e-05,  1.5219e-01,  1.0844e-01,
 7.6218e-02,  1.7016e-01,  9.2438e-02,
 4.3387e-02,  8.0141e-02, -3.2034e-02,
 9.2121e-03, -2.8742e-03, -1.5988e-03,
 9.1980e-03,  1.6983e-02,  3.3154e-03,
-2.5642e-02,  4.1607e-03,  6.9246e-03,
 3.7665e-40, -4.0391e-41, -4.0502e-41,
 2.2436e-40, -1.7190e-40,  1.6583e-40,
 1.4090e-40,  2.2914e-41,  6.7388e-41,
-8.1776e-02,  9.0814e-02,  1.0222e-01,
-3.4949e-02,  1.0266e-01,  3.6826e-02,
-8.3856e-02,  1.1102e-01,  1.1026e-01,
 1.5993e-02, -1.1626e-01, -3.0870e-01,
-3.4119e-03,  1.7638e-01, -1.9092e-01,
-1.2549e-01,  3.2538e-01, -7.9381e-02,
 3.8433e-03, -8.2530e-02,  3.2103e-02,
-1.1637e-02, -1.0371e-01,  2.3851e-02,
 2.5390e-02,  7.7085e-02,  8.9536e-02
}
,)"
R"(
{
-2.8918e-02, -8.3719e-02, -3.3026e-02,
-2.2620e-01,  2.4280e-02, -2.1254e-01,
 2.8231e-02,  3.5323e-02, -2.8425e-02,
 1.6891e-01,  3.8192e-03,  7.2794e-02,
-1.6364e-01, -4.1031e-02, -1.3141e-02,
-3.9478e-02,  1.4910e-01, -7.0978e-02,
-6.3880e-02,  9.8206e-02,  1.3163e-01,
 1.5778e-01,  1.1914e-01,  3.3277e-01,
-3.6808e-01, -5.5627e-01,  1.4401e-01,
-4.0314e-01,  3.6298e-01, -3.8212e-02,
-2.3782e-01,  2.5410e-01, -2.2334e-01,
 7.6542e-02,  9.4998e-02,  3.3399e-02,
-1.8601e-01, -1.8863e-02, -4.1835e-02,
-5.8671e-02, -8.9987e-02, -6.1069e-02,
-7.1062e-02, -9.5987e-02,  1.2318e-02,
 5.4541e-39, -1.8871e-39,  4.5048e-39,
-2.2237e-39, -5.4753e-39,  1.4395e-39,
-3.5753e-39,  6.1466e-40, -2.1567e-39,
 4.5273e-02,  1.1619e-02,  1.1379e-01,
 1.4093e-01,  1.0444e-01,  1.1283e-01,
-3.0230e-02,  3.1937e-01,  5.0541e-02,
 8.2862e-02, -3.1540e-02, -6.4833e-02,
 1.5168e-01,  1.7613e-03,  4.2690e-02,
 1.8820e-01,  4.3783e-02,  6.3473e-02,
 8.0477e-02,  1.0397e-01, -3.6337e-02,
-7.2828e-02,  6.4048e-02,  4.2476e-02,
-1.3974e-04, -2.2468e-01, -4.9189e-02,
-2.7478e-03,  8.7663e-03,  4.3870e-02,
-3.3168e-02,  1.1915e-01, -1.8083e-02,
 4.8155e-02, -4.1742e-02,  1.1251e-01,
-6.1535e-02,  5.1782e-02, -2.3494e-02,
 5.1677e-02,  1.4067e-01, -1.0377e-01,
 3.2951e-03,  1.1942e-02, -1.1775e-01,
-2.2104e-02, -8.1073e-02, -3.7509e-02,
 6.8970e-03,  1.6406e-02,  4.6923e-02,
-8.8448e-03,  2.9130e-02,  3.1024e-02,
 7.6795e-02,  4.6816e-02, -1.3204e-02,
 1.3988e-01,  1.1175e-01,  8.7121e-02,
 1.2097e-01, -3.8463e-02,  6.7387e-02,
 1.4708e-39,  1.7125e-39,  2.7764e-39,
 1.5203e-39,  1.5811e-39,  4.4921e-39,
 1.8828e-39,  1.7593e-39,  2.3774e-39,
 4.3474e-02, -4.7065e-02, -7.1999e-02,
 6.0338e-02,  3.7240e-02,  2.8802e-02,
-4.0701e-02,  1.8627e-02, -1.8181e-02,
 5.5169e-02,  1.1874e-01, -7.0475e-02,
-1.3438e-02,  1.4335e-01,  1.5180e-01,
 5.6331e-02,  7.9719e-02,  6.2691e-03,
-6.6460e-02,  2.7455e-01,  5.5916e-02,
 1.3515e-01, -3.7263e-01,  1.3463e-01,
-4.0820e-05,  3.1896e-01, -8.3871e-02,
-7.6172e-02,  6.1963e-02, -1.3804e-02,
-5.2852e-02,  1.0006e-01, -3.4106e-02,
 6.7218e-02, -3.8616e-03, -7.1788e-02,
 1.6386e-02, -1.8612e-02, -1.7354e-01,
-1.2166e-01,  1.2667e-02, -3.3852e-02,
-3.2897e-02,  1.0343e-01,  2.4924e-01,
-1.3272e-02,  1.5705e-01,  6.7731e-02,
 1.0637e-01,  1.9482e-02, -2.0655e-01,
-5.9087e-03, -7.1073e-02,  1.8723e-02,
-2.6087e-02,  1.5997e-01,  9.6264e-02,
 1.2431e-01,  1.1462e-01, -9.7197e-02,
-6.2347e-02, -4.5239e-02, -2.6443e-02,
 3.7406e-39, -4.6345e-40,  3.7971e-39,
-3.8112e-39, -3.5585e-39,  4.6938e-39,
 6.0588e-39, -4.2403e-39,  1.5311e-39,
 1.6381e-01, -6.8390e-02,  2.6527e-02,
-9.8612e-02,  2.1953e-01, -2.1886e-01,
 7.4841e-02, -1.2118e-01, -8.1700e-02,
 4.4974e-02,  7.7514e-02, -8.4620e-02,
-2.9808e-02,  2.1591e-02, -3.9502e-02,
-5.5797e-02, -6.5105e-02, -5.9860e-02,
-3.7811e-01, -2.3056e-01, -7.4491e-02,
 4.0833e-02, -2.2613e-01, -1.4986e-01,
-1.0974e-01, -6.5161e-01,  1.7546e-01,
 7.7903e-02, -1.5969e-02, -6.3040e-02,
-1.7819e-01, -7.1414e-02,  1.8451e-02,
-1.0618e-01,  3.5614e-03,  3.6719e-02,
 1.5666e-01,  3.9222e-01,  9.1678e-02,
 1.4519e-01,  5.7331e-01, -7.3466e-02,
 1.0271e-01,  1.0803e-01, -1.3150e-01,
 3.7496e-01,  1.5001e-01,  1.4727e-01,
 3.2151e-01,  1.2875e-01, -8.1645e-02,
 2.8629e-01,  1.9329e-01, -8.0009e-02,
-9.9557e-02, -2.6954e-02,  2.6042e-02,
-5.3374e-02,  1.1369e-01,  4.6503e-02,
-3.4068e-02,  9.1849e-03, -9.1420e-02,
 4.6343e-39,  4.8289e-40,  3.1694e-40,
-3.5093e-39, -4.7356e-39,  7.1265e-40,
-4.9626e-39, -2.1280e-39,  1.8542e-39,
-1.3634e-01, -5.4825e-02, -6.6125e-02,
-2.0694e-01,  1.4924e-01,  1.4028e-01,
 3.2735e-02,  7.6360e-02, -9.2541e-02,
-1.2149e-01, -7.9789e-02, -2.9591e-02,
 1.2852e-02,  1.2457e-01,  1.3081e-02,
-3.2966e-03,  1.1089e-01,  8.6461e-02,
 1.4352e-01,  5.9238e-02, -2.1140e-02,
 7.3999e-02,  2.0893e-01,  3.5512e-02,
-5.3110e-02,  3.9222e-01,  1.3103e-01,
 1.0168e-01,  1.6685e-02,  5.1616e-02,
 9.8241e-02, -1.6502e-01, -1.2586e-01,
 8.3915e-02,  7.4837e-03,  5.7355e-02,
-3.4982e-02, -1.2773e-01,  6.8213e-02,
-1.4674e-01, -3.6844e-01,  8.1546e-02,
-1.5385e-01, -7.0368e-02,  4.3894e-02,
 7.8201e-02, -1.3952e-01,  1.5154e-01,
 2.3880e-02,  1.4078e-01, -1.2906e-01,
-1.8268e-01, -1.5687e-02, -1.2588e-01,
-9.4643e-03,  1.4718e-02,  7.4932e-02,
 3.0996e-02, -1.2339e-01,  1.7452e-01,
 4.4221e-02, -1.3808e-01, -1.0205e-02,
-8.6959e-40, -3.7907e-39, -1.6020e-41,
 4.3567e-40,  1.4647e-39,  6.5692e-40,
 5.4286e-39,  8.8667e-40, -3.5047e-39,
 2.4116e-02, -9.5358e-02,  1.6468e-01,
 3.1916e-01, -2.3472e-01, -2.1644e-01,
 1.2945e-01, -1.8403e-02, -3.2247e-02,
 1.3666e-02, -3.0548e-02, -4.7635e-02,
-9.2714e-02, -2.1605e-01, -5.9464e-02,
-8.9110e-03, -3.9299e-03, -2.3289e-02,
-1.7855e-01,  9.0661e-03, -1.9142e-02,
-5.6754e-02, -5.4451e-01, -5.7664e-01,
 1.6835e-01,  2.0531e-02,  2.0812e-01,
 5.2794e-02, -9.0414e-02,  3.5560e-02,
 3.7395e-02,  5.9355e-02, -3.6676e-02,
 3.8035e-02,  6.7844e-02,  1.1042e-01,
 5.0372e-02,  6.8188e-02, -8.5353e-02,
 2.2769e-01,  5.9758e-01, -7.4568e-02,
 7.8316e-02,  8.4925e-02, -4.0400e-02,
-7.7984e-02, -2.0739e-01,  1.1736e-01,
 2.4528e-02,  2.1850e-01,  2.5639e-01,
-2.4561e-02,  8.4661e-02, -9.2191e-02,
-2.7006e-02, -7.8921e-02, -2.7124e-02,
-5.9232e-03, -2.7693e-02,  5.9524e-02,
 9.7704e-02,  9.6223e-02,  2.0432e-02,
-2.5588e-39,  5.5478e-39, -5.6209e-39,
-4.7285e-39,  4.5875e-39, -5.7483e-39,
 6.7240e-40, -3.5113e-39, -3.6246e-39,
 1.6870e-03, -2.1707e-01, -3.8895e-02,
-5.8465e-02, -5.9146e-02,  1.1936e-01,
-2.7727e-02, -9.5047e-02, -2.2627e-01,
-9.5155e-02, -7.1422e-02,  9.4611e-03,
 3.7587e-03,  1.6966e-02,  2.8839e-02,
-3.0794e-02,  1.9888e-02, -5.2541e-02,
-1.0708e-02,  3.0171e-02, -3.0473e-01,
-1.0214e-01,  4.2017e-02,  2.5568e-01,
-9.8664e-02, -5.5928e-01, -7.6876e-02,
-8.6821e-03,  4.6484e-02, -3.0836e-01,
-1.0205e-01,  6.8113e-02, -2.8059e-01,
-5.7828e-02,  2.0990e-02, -1.2843e-01,
 7.5680e-02,  1.7504e-02,  1.6278e-01,
 1.4075e-01,  2.4361e-01,  2.2737e-01,
-1.3044e-01,  8.2145e-03,  1.6344e-01,
-2.4780e-03,  1.5108e-01,  1.3313e-02,
-9.5257e-02,  6.1810e-02, -1.9386e-01,
 7.1365e-02,  1.5328e-01,  9.5848e-04,
 1.2278e-01,  7.8318e-02,  3.3400e-02,
 4.8597e-02,  6.0632e-02, -5.7238e-02,
 3.2522e-02,  4.5926e-02, -9.5566e-02,
 1.0844e-39, -3.2490e-39, -2.6904e-39,
-3.0517e-39,  4.7535e-39,  4.3440e-39,
-1.3996e-39,  4.5201e-39, -3.6165e-39,
-5.6164e-02,  1.0353e-01,  6.6228e-02,
 8.2147e-02,  4.7827e-01,  1.2004e-01,
-6.8150e-02,  1.8340e-01,  2.2113e-01,
 1.0580e-05, -2.0949e-01, -1.0358e-01,
 1.6206e-01,  1.2538e-01, -1.3104e-01,
 1.3700e-01,  2.9282e-02, -8.7020e-02,
 4.5467e-39,  5.9787e-39,  2.6105e-39,
-1.2670e-39,  2.9513e-39, -1.0811e-39,
-3.9129e-39, -1.8499e-39,  2.9297e-39,
 5.7414e-39,  5.5907e-39,  5.5702e-39,
 5.9004e-39,  5.7585e-39,  6.3188e-39,
 5.7395e-39,  5.6146e-39,  5.6451e-39,
-7.3964e-39, -6.3330e-39, -5.5236e-39,
-7.5172e-39, -5.8828e-39, -3.7555e-39,
-6.9528e-39, -7.7656e-39, -5.5115e-39,
-7.9031e-39, -7.8200e-39, -7.7914e-39,
-7.4570e-39, -7.6413e-39, -7.9054e-39,
-7.3437e-39, -6.7956e-39, -7.0789e-39,
-3.6774e-40,  1.3572e-40,  3.0250e-40,
-4.1792e-40, -4.6240e-40,  2.2528e-40,
-5.2143e-40, -5.6847e-40, -4.2768e-40,
-4.0128e-39,  1.3485e-39,  1.3436e-39,
 1.5337e-39, -3.9186e-39,  1.2120e-39,
 1.2992e-39,  1.5671e-39,  1.5659e-39,
-4.6533e-39, -4.7029e-39, -6.0334e-39,
-5.1157e-39, -5.3257e-39, -5.8595e-39,
-4.3046e-39, -4.4391e-39, -5.0039e-39,
-1.0025e-39, -1.0145e-39, -8.6762e-40,
-1.0282e-39, -1.0939e-39, -9.4134e-40,
-1.1868e-39, -1.2133e-39, -5.4261e-40
}
,)"
R"(
{
-1.2633e-01,  2.7332e-01, -4.6674e-01,
-9.4537e-03,  9.6797e-02, -6.4975e-01,
 1.8103e-02,  2.7190e-03,  2.3888e-01,
 4.8553e-02, -8.7297e-02,  1.8415e-01,
 3.1194e-02, -7.2899e-02, -8.1835e-02,
 7.1639e-02, -3.1455e-02, -6.2866e-02,
-2.1413e-02,  4.6066e-02,  9.2372e-02,
 1.5761e-01, -1.0352e-01, -3.4808e-01,
 2.3715e-02,  1.6453e-01, -1.3699e-01,
 1.1705e-01, -1.6882e-02,  1.2575e-01,
-2.9834e-02, -1.1558e-01,  4.7318e-01,
 3.5301e-02,  1.1246e-01,  3.5038e-03,
 1.5837e-01, -2.9968e-01,  1.6094e-01,
 4.0562e-02, -1.6329e-01, -3.7023e-02,
-3.9991e-02,  1.7001e-01, -2.7735e-03,
 8.8139e-02, -2.4828e-01,  5.5751e-04,
-1.3871e-01, -2.4839e-01,  1.7996e-03,
-1.1670e-01,  3.3651e-02, -2.9559e-02,
 3.8572e-03,  3.7329e-02,  4.7511e-02,
-7.8848e-02,  1.2844e-01,  9.2677e-02,
-8.5041e-02,  5.7212e-02, -1.0415e-02,
-3.2462e-39,  2.3003e-39,  4.9676e-39,
-3.9261e-39, -6.8290e-40,  5.9119e-39,
-4.1242e-39, -1.1996e-39,  3.8436e-39,
-2.3243e-02, -2.2525e-02,  3.9668e-02,
-1.1210e-01, -2.3892e-01,  1.6431e-01,
-1.3998e-01, -1.5857e-01, -1.5625e-01,
-1.7634e-02, -3.9174e-02, -9.0936e-03,
-3.9428e-03, -1.6411e-02,  2.6484e-03,
 1.1376e-02, -2.9057e-03,  6.3382e-02,
 4.8930e-02,  9.1298e-02,  1.8195e-02,
-6.3365e-02, -1.5407e-01,  8.1543e-02,
 4.9919e-02,  1.6852e-01,  4.4053e-02,
-4.8682e-02, -7.3614e-02, -6.9206e-03,
-4.8193e-02, -2.3704e-01, -8.3394e-03,
 5.6024e-02,  3.7845e-01, -2.4550e-02,
 5.2050e-02,  2.2027e-01, -4.1328e-02,
-6.6327e-02,  1.0450e-01,  1.7058e-02,
-1.2047e-01,  5.2494e-02, -1.8018e-02,
 5.4807e-02,  1.1177e-01,  2.3511e-02,
 6.0413e-03, -3.2457e-02,  7.6611e-02,
-2.1276e-02,  3.0054e-02,  5.0752e-02,
 7.5556e-02,  2.5734e-02, -6.0634e-02,
 1.2201e-01, -4.1533e-01,  2.7634e-02,
 4.5560e-01,  3.2832e-01,  2.6277e-02,
 1.9889e-39,  3.8337e-39,  4.0170e-39,
 1.5149e-39,  3.6456e-39,  4.0474e-39,
 1.1508e-39,  2.7381e-39,  3.8673e-39,
-7.9206e-02, -2.0763e-02, -2.4842e-01,
-6.5777e-02, -1.8446e-01,  2.6178e-01,
-1.7908e-02, -2.3039e-01, -3.5767e-01,
 1.0324e-02,  1.3610e-01,  8.6519e-02,
 1.3499e-01,  3.1933e-02,  9.1822e-03,
-3.6017e-02, -2.2056e-01, -2.3258e-01,
-7.6185e-02, -2.8981e-01, -1.1816e-01,
-9.9048e-02,  5.3879e-02, -1.7351e-01,
-2.1874e-01, -1.2109e-01, -3.1457e-01,
 5.1576e-02, -2.5656e-02,  4.6789e-02,
 7.6286e-02,  6.0126e-01, -2.5925e-01,
-5.3443e-02, -3.3656e-01,  4.7585e-01,
-4.7442e-02, -5.1580e-02, -8.5216e-02,
-1.0600e-01, -1.3859e-01, -3.1484e-01,
 2.1454e-01, -1.1851e-01, -7.6614e-02,
-7.8873e-03, -7.0275e-02, -1.0958e-01,
-8.0654e-02,  1.3946e-01,  2.5292e-01,
 1.3254e-03, -6.7372e-02, -2.6429e-01,
-8.2344e-02,  1.2388e-01,  5.2930e-02,
 8.3665e-02,  3.9729e-01,  4.7687e-02,
-4.4502e-02, -8.3105e-02, -1.6430e-01,
 1.2825e-39,  1.7532e-39,  2.1774e-39,
-2.1331e-39, -2.1826e-39, -1.0009e-39,
 3.7081e-39,  2.0015e-39, -5.8349e-40,
-3.5278e-02,  6.5211e-02, -5.4199e-03,
 8.3961e-02,  3.1410e-02,  4.4510e-02,
-5.4905e-02,  4.0727e-02, -1.5710e-02,
 1.0813e-01,  8.2043e-03,  4.1303e-02,
 1.3405e-01,  1.4150e-01,  7.2155e-02,
 3.3942e-02, -4.7781e-02,  1.6095e-01,
-1.4266e-01, -2.5283e-02,  6.4043e-03,
-1.8699e-02,  1.0895e-01, -2.1497e-02,
 5.5074e-02,  1.7031e-02,  1.0572e-01,
 7.3199e-04,  1.0813e-01, -9.0280e-05,
 1.4808e-01,  2.5436e-01, -1.3749e-01,
 2.2936e-02, -7.9733e-02, -2.2360e-01,
 6.0406e-02, -1.2874e-01, -7.4692e-02,
-1.3216e-01, -9.9889e-03,  2.7608e-03,
-1.1412e-01, -5.1312e-02, -1.7196e-02,
-2.2800e-02, -1.2112e-01, -9.3855e-03,
 3.6905e-02,  1.0049e-01,  9.0602e-03,
-7.3200e-02,  1.0628e-01, -4.8218e-02,
-4.6525e-02,  6.0314e-02, -3.6467e-03,
-8.0943e-02,  2.5461e-01,  1.5461e-01,
-5.7708e-02, -5.7823e-02,  5.4042e-02,
 3.8847e-39,  3.5806e-39,  4.1610e-39,
 3.9082e-39,  4.1898e-39,  4.1926e-39,
 4.1200e-39,  4.3759e-39,  4.3977e-39,
-3.3576e-01,  9.5443e-02,  2.7804e-02,
-2.3834e-01, -7.2650e-01, -1.2229e-01,
 1.0380e-01,  1.9520e-01,  3.4571e-02,
-3.7291e-02,  7.6216e-02,  8.6171e-02,
-1.6324e-01, -8.6759e-03,  4.3038e-02,
-3.4364e-02, -7.2777e-03,  3.7451e-02,
 1.8826e-01,  1.6387e-01, -3.4750e-02,
-2.0203e-01,  2.4170e-01,  9.0358e-05,
-1.3049e-01,  9.6855e-02, -1.6737e-03,
-6.3782e-02,  7.1413e-02, -6.5077e-02,
-1.5262e-01,  4.3261e-01, -8.4224e-02,
 6.4632e-02,  1.0553e-01, -1.5274e-01,
 4.4294e-05,  8.6239e-02,  5.7537e-03,
-5.7633e-01, -5.0076e-03, -5.2298e-02,
 1.8556e-01, -1.1332e-02, -2.7010e-02,
 1.6155e-01, -3.0337e-02, -9.6808e-03,
-2.8404e-01, -2.7625e-02,  1.6058e-02,
 5.7937e-02, -6.6464e-02,  1.1096e-02,
 7.8268e-02,  8.6122e-02,  2.9298e-02,
 6.4696e-02,  2.0285e-01,  4.3660e-02,
 1.5339e-01, -3.7650e-02,  7.1438e-03,
-8.9058e-40, -3.6429e-39, -4.7562e-39,
 8.3914e-40, -2.8054e-39, -3.6702e-39,
 4.3666e-39, -1.0602e-39, -3.0369e-39,
 7.2731e-02, -1.0227e-01, -1.9583e-02,
-1.7466e-02, -2.0097e-01,  9.3108e-02,
 6.5196e-02, -1.1880e-01, -3.5152e-03,
-5.6533e-02,  6.2109e-02,  5.2029e-02,
 5.7971e-02,  5.1577e-02,  6.6318e-02,
-2.1669e-03,  7.7274e-02, -4.0609e-02,
 2.8531e-02, -8.3960e-02,  1.3615e-02,
-1.1151e-02, -1.4162e-03,  5.6661e-02,
-8.0954e-02, -1.0600e-01,  4.3276e-02,
 7.6762e-04,  3.1437e-02, -6.1084e-02,
-8.1119e-02,  2.1406e-01,  6.0836e-02,
 4.8105e-02, -1.6263e-01,  9.2555e-03,
 1.1060e-01, -2.1090e-01,  1.6435e-01,
-1.0248e-01, -1.1884e-01, -7.9929e-02,
 5.9980e-02,  1.0271e-01, -1.1891e-02,
-7.5044e-02, -2.3655e-02, -5.2865e-02,
 2.1542e-02,  2.7305e-04,  1.3508e-01,
-1.2317e-02,  9.0742e-02, -3.0079e-03,
-9.9020e-02,  1.5578e-01, -2.1482e-03,
-8.9029e-02,  1.8470e-01,  3.7571e-02,
-2.0394e-01, -1.3735e-01,  2.9648e-02,
-4.3016e-40, -7.3591e-40, -7.3773e-40,
-4.1239e-40, -8.6029e-41, -6.9504e-42,
-7.5082e-40,  1.2975e-40,  2.1462e-40,
-1.8967e-02, -1.4903e-01,  8.1452e-02,
 1.2099e-01, -2.5524e-02,  1.3285e-02,
-1.3780e-01, -5.3359e-02, -3.1310e-02,
-1.8984e-02,  4.1962e-02,  1.0186e-01,
-1.0823e-01,  1.1079e-01,  7.8613e-02,
-1.4521e-01, -7.7509e-02,  1.8768e-02,
 5.0613e-03, -3.0459e-02, -6.3055e-02,
 4.4540e-02,  2.0135e-01,  9.6351e-02,
-1.9495e-02, -1.2314e-01,  1.1720e-02,
 2.1739e-02,  5.2098e-02, -4.0453e-02,
-9.9983e-02,  4.7578e-02, -2.7862e-02,
-8.6565e-02,  1.5241e-01, -4.0462e-02,
 4.0458e-02, -1.2871e-01, -4.3491e-02,
 9.8981e-02, -1.3637e-01,  2.0092e-02,
 1.5626e-01, -8.4550e-04, -2.5701e-02,
 1.8511e-02, -1.0257e-01, -7.3238e-02,
-3.9802e-02, -1.6120e-02, -7.4068e-04,
-1.1377e-02,  9.7975e-03, -9.0342e-02,
-6.7152e-02,  1.0208e-01,  2.5234e-02,
-4.3687e-02,  2.5334e-01,  9.2712e-02,
 3.7702e-01,  4.1450e-02,  1.9934e-02,
-5.4201e-39, -6.7158e-39, -7.5025e-39,
-5.2548e-39, -6.4829e-39, -7.2782e-39,
-4.9999e-39, -5.9599e-39, -6.0469e-39,
 3.5890e-02, -7.3738e-02,  9.8899e-02,
 3.3312e-02,  5.8231e-02, -2.1348e-01,
 8.6289e-02,  5.0837e-02, -6.5613e-02,
 7.0208e-02,  4.1424e-02, -6.0761e-02,
 4.4654e-02, -3.3590e-02, -5.3044e-02,
 1.2319e-01, -4.4666e-02, -8.8193e-02,
-9.0463e-02, -3.0083e-02,  6.8075e-02,
 4.2531e-02,  4.3248e-01,  1.3480e-01,
 9.2389e-02,  1.3683e-01, -2.6092e-01,
 2.8925e-02,  2.3317e-01,  7.8128e-02,
 6.3444e-02,  1.6291e-01, -3.8727e-03,
 6.9107e-02,  6.8477e-03,  3.9528e-01,
 3.8471e-02,  3.0745e-02,  2.8446e-02,
 1.0625e-02, -2.4006e-01, -1.2490e-01,
-1.3002e-01,  2.0025e-01,  4.7618e-02,
-3.9705e-02, -1.2017e-02, -9.8790e-02,
-1.2798e-02, -2.7540e-01, -1.5138e-01,
-1.0290e-01,  5.0112e-02, -1.7391e-01,
-9.7079e-02, -2.2350e-03, -5.9211e-02,
-2.4728e-01,  4.3353e-01, -1.9306e-01,
-1.8039e-01,  1.2689e-01,  5.2103e-02,
-4.5547e-39, -7.8040e-39,  4.1196e-39,
 1.5214e-39,  9.3494e-40, -3.9058e-39,
 7.8718e-39,  7.1728e-39,  5.3609e-39
}
,)"
R"(
{
-9.4505e-02, -7.0477e-02, -1.5792e-04,
-2.3475e-01,  5.8849e-02, -6.8161e-02,
 7.0658e-03, -1.0276e-01,  7.2471e-02,
-7.3820e-03, -3.0740e-02, -1.1131e-01,
 2.8429e-02, -3.5750e-01, -8.4683e-02,
-5.0210e-02, -3.1096e-03, -2.3730e-02,
 4.5756e-02, -3.6724e-01, -7.6317e-02,
 3.8467e-01,  5.5354e-02,  1.6943e-01,
-4.9403e-02,  7.4709e-02, -3.0550e-02,
-7.5324e-03, -1.6910e-01, -1.6103e-01,
 4.6314e-02,  1.2912e-01, -3.0488e-02,
 2.6388e-02,  5.6925e-02,  6.4396e-02,
 3.7748e-03, -2.1310e-02,  1.1410e-01,
-7.0164e-03,  1.8228e-02, -2.5920e-01,
 6.8416e-02,  1.3998e-01,  1.3290e-01,
-3.8861e-02,  8.9898e-02, -3.6631e-03,
 3.5528e-02,  1.1249e-01,  3.7018e-02,
-6.2334e-02, -4.8470e-02, -4.4094e-02,
 3.1574e-02, -1.2162e-01,  1.9669e-01,
-4.6605e-03,  1.1887e-02, -1.1958e-01,
-1.0736e-01,  6.0131e-02, -1.2829e-02,
 2.1305e-01, -8.4750e-02, -2.7028e-02,
-3.0351e-01, -6.4246e-03, -7.9128e-02,
 1.3081e-01,  9.5878e-02,  1.6193e-02,
-5.8335e-02, -5.5968e-02, -2.6284e-03,
-7.2218e-02, -1.1661e-02,  1.9413e-03,
-1.6043e-01,  1.1388e-01, -3.6473e-02,
-2.4077e-02,  1.2210e-01,  1.5531e-02,
 1.5074e-01, -4.5545e-01,  6.1004e-02,
-6.3948e-02,  3.9804e-02, -4.8822e-04,
 1.3135e-01,  9.2392e-02,  8.8914e-02,
 1.2941e-01, -3.6052e-01,  3.9571e-02,
-2.4838e-02,  7.0425e-02, -1.9016e-02,
 2.7629e-02, -7.0648e-02, -2.6838e-02,
-2.1844e-02, -9.6184e-02, -3.3611e-02,
 8.5938e-02,  5.2663e-02,  2.2938e-02,
-6.9909e-03, -3.9627e-03, -6.5162e-02,
-4.9296e-03, -4.0383e-02,  6.7670e-01,
 1.5251e-02,  2.1000e-01, -1.9137e-01,
 2.2825e-02,  1.6640e-02,  3.8147e-02,
 7.1902e-02, -4.9821e-02, -6.5592e-03,
 1.5826e-02,  2.1626e-02,  1.1646e-02,
 1.5180e-02,  1.5664e-01,  9.8696e-03,
-7.2901e-02, -2.1818e-01,  9.2465e-02,
 6.4349e-02,  6.0290e-02, -2.1094e-02,
 2.0633e-02,  4.8808e-02,  1.4080e-02,
 4.8083e-02, -1.5979e-01, -5.3634e-02,
 6.5004e-02,  7.0317e-02,  1.9117e-02,
-4.3048e-02,  5.9627e-02, -1.5068e-02,
 1.8861e-01, -2.6868e-01,  1.2789e-03,
 1.1273e-01, -2.7796e-01,  4.9841e-02,
 4.9008e-03,  1.8241e-02,  4.3449e-02,
 2.1420e-02, -1.0299e-01, -1.6235e-01,
-1.9300e-02, -1.5121e-02,  2.0616e-03,
-2.7591e-01,  3.9622e-02, -5.0492e-02,
 1.1866e-01,  5.5502e-01, -2.3622e-02,
-6.1204e-03, -7.4778e-03,  6.7961e-03,
 2.4215e-02,  2.1643e-03,  1.1442e-01,
 7.5326e-02,  1.4455e-01,  8.0497e-02,
 6.6115e-02,  2.9762e-02,  2.8680e-02,
 3.7784e-03, -2.2769e-02,  2.4529e-02,
-1.1441e-02,  9.8463e-02, -1.2761e-02,
 1.0642e-02,  5.2871e-02,  1.9650e-01,
-2.2225e-02,  3.1504e-02,  8.5645e-03,
 4.9125e-02,  1.4439e-01,  8.4573e-02,
 1.0103e-02,  1.9097e-02,  4.5579e-03,
-2.5773e-02, -4.0984e-02, -1.5402e-01,
 5.3050e-02,  1.5509e-01, -1.9040e-01,
 3.7700e-02,  1.0632e-01, -2.2520e-02,
-5.6582e-02, -4.6040e-02, -5.7562e-03,
-3.4924e-01,  3.2933e-01,  5.5211e-02,
 2.3230e-02,  8.5108e-02,  3.7448e-02,
 1.4266e-02, -7.2016e-02,  4.5252e-03,
-7.0246e-02,  3.9142e-01, -1.9216e-02,
 2.0536e-01, -3.5615e-01,  3.8009e-02,
 1.2252e-02, -5.7966e-02,  9.2672e-02,
 2.4225e-02, -1.0186e-01, -1.4219e-01,
-2.8815e-02,  1.3088e-02, -2.6031e-03,
-6.2341e-02, -1.1216e-01, -7.2122e-02,
 1.1812e-01,  4.3493e-01,  4.3593e-02,
-1.3524e-02,  4.8679e-03, -1.0598e-02,
 3.4904e-02,  5.5813e-02,  4.6811e-02,
 8.0928e-02,  7.6607e-02,  6.3968e-02,
 5.4647e-02,  2.8693e-02,  2.1957e-02,
-8.2725e-03,  5.4668e-02, -3.0533e-02,
-9.3953e-03,  1.5874e-01, -3.6093e-01,
 5.6412e-03,  1.8977e-02,  2.0088e-01,
-1.9414e-02,  1.9088e-02,  1.4504e-02,
 5.8462e-02,  6.2645e-02,  4.9884e-02,
 6.6913e-03,  4.3639e-02,  1.5139e-02,
-2.1897e-02, -1.1436e-01, -5.0838e-02,
 7.1176e-02,  8.4667e-02, -1.4480e-01,
 3.7676e-02,  1.0840e-01, -2.6417e-02,
-4.7584e-02, -4.0524e-02,  6.3032e-03,
-2.4822e-01,  2.4635e-01,  5.5942e-03,
-1.3347e-02,  1.0515e-01,  4.2549e-02,
-1.2380e-01,  4.1074e-02,  1.2608e-02,
-1.2042e-01,  2.9516e-01,  2.8380e-03,
 5.1930e-01, -1.6498e-01,  5.7152e-02,
-6.5519e-02,  1.1001e-01,  2.8943e-02,
 1.0854e-01, -6.0107e-02, -1.6730e-01,
-4.4417e-02,  3.4347e-02, -3.3756e-02,
 2.0694e-01,  3.3047e-01, -9.4497e-02,
-2.1977e-01,  4.6614e-02,  1.2201e-01,
-2.9541e-02,  1.8900e-01, -1.8391e-01,
 2.0064e-02, -3.2480e-02, -8.9041e-03,
-5.6385e-02, -6.4531e-02,  1.2879e-02,
-3.2499e-02,  1.0883e-02,  7.3564e-03,
 1.9828e-02, -2.3278e-01, -4.3789e-03,
 9.7669e-02,  1.3008e-01, -1.0405e-01,
 2.2618e-02, -2.5495e-01, -1.0718e-01,
 4.3524e-02, -7.3127e-02,  8.2424e-02,
-5.0193e-02,  4.0634e-03,  4.0696e-02,
 2.7419e-02,  1.8353e-01,  9.2117e-02,
-7.4918e-02,  1.0602e-01, -3.4752e-02,
-1.3331e-01, -2.9583e-02, -5.2197e-03,
-3.7852e-02,  1.5998e-01,  1.5078e-03,
-5.6512e-02,  1.3378e-01,  1.4512e-02,
 4.5255e-02,  2.4702e-01, -2.4848e-02,
-1.7526e-01,  1.5532e-01,  8.6686e-02,
 3.1486e-02, -2.3247e-02,  9.7320e-03,
-5.2106e-01,  4.7937e-02,  4.1614e-02,
 5.5436e-02, -2.0432e-01,  1.2444e-02,
-5.6792e-02, -5.5632e-02,  5.7612e-02,
-6.0248e-04,  4.9770e-02, -6.7956e-02,
 1.3389e-02, -9.4141e-03, -7.3497e-03,
-4.6361e-01,  2.7450e-01, -8.2210e-02,
-2.6737e-01, -6.6114e-02,  6.3568e-02,
 1.6910e-02,  1.4456e-01, -9.0081e-02,
 8.8278e-03,  2.1776e-02,  8.7710e-03,
-2.3378e-02, -4.3907e-02, -3.6751e-02,
-2.4694e-03, -6.0419e-03,  3.0840e-02,
-1.6968e-02, -8.2266e-02, -1.0049e-01,
 3.4429e-02,  1.0960e-01,  3.8355e-01,
-4.0301e-04, -3.1089e-02, -2.1373e-02,
-2.4172e-02,  4.6432e-02,  8.0742e-03,
-2.3134e-02,  1.7789e-02,  2.7136e-02,
 3.0729e-02,  6.9008e-03,  1.2822e-02,
 3.5043e-02, -6.1749e-02, -1.2565e-02,
-1.0354e-02, -2.6515e-03,  4.5632e-03,
-5.9818e-02, -9.7686e-04, -6.6467e-03,
-5.0833e-01,  1.8474e-02,  1.3598e-02,
 3.6287e-01,  1.3698e-01, -1.2806e-02,
-2.8618e-02, -2.9128e-02,  2.9855e-02,
 8.1243e-02,  4.7414e-02, -4.7434e-02,
-3.3738e-02, -3.4926e-01,  1.7786e-02,
 1.0056e-01, -5.7937e-02, -1.8308e-02,
 1.8214e-02, -1.9519e-01,  2.2152e-02,
-7.3543e-02,  2.0786e-01, -5.8196e-02,
 3.9396e-02, -4.5349e-02,  1.5748e-02,
-5.4604e-03,  4.5777e-01,  1.7295e-01,
-2.0570e-01, -3.0970e-01, -1.9075e-01,
 7.6751e-02, -1.3099e-01,  6.1278e-02,
 6.0222e-02,  5.4418e-02,  1.2259e-01,
 3.2160e-02,  8.5146e-03,  3.4578e-02,
-5.4391e-02, -2.5285e-02,  1.0251e-02,
-3.2763e-02,  7.9163e-02, -7.5136e-02,
 1.8545e-02, -2.1972e-02,  1.3887e+00,
-1.2402e-03, -2.5679e-01,  7.2392e-02,
 4.9692e-03,  1.7034e-02,  4.7043e-02,
 1.2093e-02, -3.1230e-02, -8.2613e-03,
-7.8701e-03, -2.3516e-03, -7.2487e-04,
 6.8495e-02, -5.2837e-02, -2.2482e-01,
 1.3259e-02,  4.8009e-01, -4.0940e-02,
-4.1547e-02, -2.8753e-02, -5.2579e-03,
-1.7152e-01, -3.3676e-02,  1.5080e-02,
 8.6014e-02,  7.9239e-02,  4.2196e-02,
-9.2870e-02, -1.5913e-02, -6.5804e-03,
 4.0364e-02,  2.4914e-02, -1.4638e-02,
 8.8705e-03,  2.8037e-01,  3.9890e-02,
 1.1638e-01,  2.9467e-01, -4.3518e-03,
 7.1091e-02, -2.2378e-01,  4.7315e-02,
 3.8006e-02, -2.0246e-01, -3.8679e-02,
-5.8004e-02,  5.8991e-02, -6.2149e-03,
-1.3034e-01,  1.5540e-01, -5.2558e-02,
 8.1594e-02,  3.5570e-01,  2.1220e-02,
 1.4977e-02,  2.4493e-03, -4.0627e-02,
 1.1402e-01,  6.6962e-02,  1.1150e-01,
 1.1824e-01,  1.1492e-01,  1.1219e-01,
 6.6067e-02,  6.9639e-02, -8.1836e-02,
-2.7144e-02,  1.4677e-01, -5.9261e-02,
 4.4573e-03,  2.6235e-01, -7.4379e-01,
-8.3569e-03,  9.4465e-02, -6.5653e-03,
 2.1095e-02, -1.8853e-02,  6.7972e-02,
 1.2957e-01,  3.0122e-02, -1.0061e-02,
-3.4832e-02,  8.5404e-02,  5.7663e-02,
-5.0400e-02, -1.2050e-01, -2.3344e-01,
 1.4977e-01,  7.8806e-02,  6.0771e-03,
 5.6483e-02,  6.3927e-02, -5.8376e-03,
-2.8124e-01,  5.2581e-02, -1.3918e-04,
-1.4341e-01,  3.6558e-01,  4.7332e-02,
-3.9089e-02,  8.4188e-02,  2.7058e-02
}
};
)"
R"(
__constant float biasL[8][8] = 
{
{
 7.2678e-02,  8.5350e-03,  5.0400e-02,  2.6268e-02,  6.2434e-02, 1.0483e-01, -7.1650e-39,  1.0062e-01
}
,
{
-4.9844e-39, -1.8567e-39,  6.0627e-04, -1.9234e-38,  1.8331e-02, -1.1364e-01, -8.3962e-03, -1.7372e-04
}
,
{
-0.0091, -0.0055,  0.0237,  0.0093, -0.0479,  0.0188, -0.0034,  0.0399
}
,
{
 6.5694e-03, -2.2259e-01, -1.1226e-02, -8.0327e-02, -1.0615e-36, 1.0402e-02,  7.6246e-03, -6.5940e-02
}
,
{
 5.0711e-02,  7.1911e-02,  2.5293e-02, -1.5608e-02,  5.3835e-02, -1.6967e-38,  2.2243e-02,  3.2742e-02
}
,
{
 1.5629e-02,  2.9703e-02,  2.6412e-02,  1.2301e-02,  1.8654e-01, -7.2260e-03,  2.4613e-02, -3.1853e-38
}
,
{
-0.0030, -0.0123,  0.0348,  0.0277, -0.0152,  0.0005, -0.0124, -0.0209
}
,
{
7.4856e-03, 7.2931e-04, 8.3015e-03, 6.4820e-03, 2.4008e-04, 7.0377e-06, 1.7948e-03, 8.9869e-03
}
};

__constant float kernelsL10[4 * 8] = 
{
 0.4240,  0.4165,
 0.1648,  0.1909,
-0.0985, -0.4455,
 0.4639, -0.0533,
-0.1368,  0.4413,
 0.2539,  0.3294,
 0.2458, -0.3256,
-0.0479,  0.3200,
-0.3977, -0.0422,
-0.2736,  0.1053,
 0.3902,  0.0594,
-0.0721, -0.2988,
 0.0495,  0.1309,
-0.1703,  0.0033,
 0.3061,  0.1827,
 0.2443, -0.1259
};)") + kernelFunction
,
std::string(
R"(#define RELU(x) fmax(x, 0.0f)

__constant sampler_t samplerN = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant float kernelsL1[9 * 8] = 
{
-0.0461,  0.1274,  0.2976,
-0.0393, -0.1251,  0.2527,
 0.0791,  0.0600, -0.0303,
-0.0520, -0.5039, -0.3305,
-0.0115,  0.0456,  0.4370,
 0.0601,  0.0780,  0.3106,
-0.0017, -0.0018, -0.0017,
-0.0017, -0.0018, -0.0018,
-0.0017, -0.0017, -0.0017,
 0.2666,  0.1687,  0.2303,
-0.1901,  0.3825,  0.3024,
 0.1811,  0.0581,  0.2080,
-0.1246,  0.0155, -0.4075,
 0.1156,  0.5929,  0.1449,
-0.1080, -0.0171, -0.0516,
-0.0817,  0.2247,  0.0472,
 0.0394,  0.1085,  0.1435,
-0.0480, -0.0135, -0.0606,
-0.0083,  0.2045,  0.1056,
-0.2239,  0.2823, -0.1926,
 0.2581,  0.1362, -0.1914,
-0.0833,  0.0702,  0.0234,
 0.3616,  0.3789, -0.1840,
 0.0128,  0.1347, -0.0187
};

__constant float biasL1[8] = 
{
-0.1329, -0.0431, -0.0031, -0.0129,  0.2294, -0.2595, -0.2370, -0.0499
};
)"
R"(
__constant float kernelsL[8][9 * 8 * 8] = 
{
{
 1.4090e-01, -1.8985e-02, -6.8589e-02,
 6.6491e-02,  1.4360e-02,  8.5223e-02,
 1.8782e-01,  9.8042e-02, -3.4558e-02,
 2.5606e-01,  2.2027e-01,  2.7603e-01,
 1.9424e-01,  3.4537e-02,  9.5975e-02,
 1.1223e-02, -4.3377e-01, -1.4760e-01,
-3.4293e-40, -5.5421e-40, -4.4763e-41,
-6.3322e-40, -3.1495e-40, -7.8264e-41,
-1.5375e-40, -3.3656e-40,  5.2441e-40,
 1.2413e-01,  1.5682e-01,  1.1465e-01,
 1.6683e-02,  7.8382e-02,  1.0110e-01,
 1.4902e-01,  1.3608e-01,  1.1674e-01,
-6.5160e-02,  7.7748e-02,  2.1773e-02,
 2.0652e-02,  2.7245e-01,  1.0297e-01,
-2.0953e-02,  6.1685e-02,  4.4128e-02,
 6.1538e-02, -1.9746e-02, -1.2785e-02,
 2.5931e-02,  1.2740e-01,  9.0033e-02,
 8.6448e-02,  2.0684e-01,  9.8063e-02,
-7.8384e-03,  6.3277e-02,  7.6751e-03,
 3.5956e-02,  1.0555e-01,  4.2728e-02,
 7.1578e-02,  1.3253e-01,  1.1171e-01,
-2.7538e-02,  1.5836e-01,  1.0014e-01,
-4.9113e-02,  1.6911e-01,  2.7329e-01,
 7.9170e-03,  9.5440e-02,  1.3922e-01,
 8.0151e-02,  4.3438e-02,  5.5314e-02,
 3.4896e-02,  1.6816e-01, -4.5783e-03,
-1.4579e-03,  2.0493e-01,  2.6238e-02,
 2.6499e-02,  3.9490e-01, -1.1582e-02,
 3.5790e-01,  1.4317e-01, -2.1775e-01,
 4.1794e-03, -3.2513e-01, -1.6729e-01,
 3.4040e-41, -6.2960e-42, -1.0067e-40,
 5.5978e-41, -1.2353e-40, -1.1347e-40,
 5.4572e-40, -6.4384e-40, -4.1234e-40,
-9.3690e-02,  1.7765e-01,  1.1275e-01,
 9.1159e-03,  1.7375e-01,  1.1427e-01,
-7.8385e-02,  1.5658e-01, -3.8399e-02,
-1.0756e-01,  5.9943e-02, -6.7273e-02,
-1.1117e-01,  1.5267e-01,  1.1563e-01,
-1.2964e-01, -3.8604e-02, -2.4532e-02,
 1.6324e-02,  1.3112e-01,  6.1679e-03,
-7.7703e-03,  2.6311e-01,  8.9427e-02,
-2.8948e-02,  1.9341e-01,  4.4339e-02,
 6.4559e-03, -6.8885e-02,  1.1481e-01,
-1.0665e-01,  3.8613e-02,  7.0410e-02,
-6.1680e-02, -1.7374e-02,  9.5475e-03,
-4.0081e-02, -3.1549e-02,  2.8311e-01,
-1.2178e-01, -1.3848e-01,  1.7416e-01,
-8.1756e-02, -1.7718e-01,  7.9533e-02,
-3.1299e-03, -3.2305e-03, -3.2094e-03,
-3.1548e-03, -3.2553e-03, -3.2453e-03,
-3.1459e-03, -3.2278e-03, -3.2076e-03,
-3.6554e-05, -3.6715e-05, -3.1284e-05,
-1.4927e-05, -1.4357e-05, -1.2185e-05,
-1.5771e-09, -1.1439e-09, -6.4952e-10,
 3.7723e-40,  4.9166e-40, -2.1946e-40,
-4.7599e-40, -4.3356e-40, -8.3928e-41,
 2.6127e-40,  4.8634e-40,  2.7720e-40,
-5.4972e-03, -5.6409e-03, -5.6919e-03,
-5.5818e-03, -5.7079e-03, -5.7542e-03,
-5.6338e-03, -5.7437e-03, -5.7600e-03,
-3.7940e-03, -3.8853e-03, -3.8693e-03,
-3.8995e-03, -3.9616e-03, -3.8945e-03,
-3.8438e-03, -3.9156e-03, -3.8269e-03,
-7.2342e-05, -7.8682e-05, -4.7701e-05,
-1.1126e-04, -1.1918e-04, -7.8931e-05,
-1.1644e-04, -1.2418e-04, -8.2350e-05,
-2.3881e-04, -3.7971e-04, -3.9448e-04,
-2.4112e-04, -3.8395e-04, -4.0189e-04,
-2.3451e-04, -3.7525e-04, -3.9222e-04,
-3.9853e-03, -4.0748e-03, -4.1134e-03,
-4.0685e-03, -4.1456e-03, -4.1548e-03,
-4.0547e-03, -4.1388e-03, -4.1357e-03,
 5.3008e-02,  2.2252e-02, -7.1158e-02,
-6.6411e-02, -3.0015e-02, -2.2526e-02,
 1.2259e-01, -6.2488e-02,  5.6190e-02,
 1.5981e-02, -7.6832e-02,  1.7908e-02,
 2.7618e-01,  5.4054e-02,  8.7282e-02,
 1.5212e-02, -1.1097e-01, -2.2265e-02,
-6.8532e-41, -6.0539e-40,  4.6269e-40,
-2.9221e-40, -3.8468e-40, -4.6656e-40,
 6.4572e-40, -6.1625e-40,  6.4545e-40,
 3.5920e-02,  9.0955e-02, -1.7626e-02,
 4.7826e-02,  1.8832e-01, -4.4043e-02,
-3.8405e-02,  5.9176e-02,  6.8182e-02,
 3.7657e-03,  2.6441e-02, -2.5585e-01,
 1.0969e-01,  2.3914e-01,  3.5120e-02,
-1.6252e-01,  3.4371e-02, -2.7501e-01,
 4.9289e-02,  2.2088e-02, -1.4588e-02,
 1.6384e-01, -8.1421e-03, -6.9613e-02,
 1.0820e-01,  1.1137e-01,  7.2648e-03,
 1.5243e-01,  1.3659e-01,  2.7553e-02,
 1.3966e-01,  1.1019e-01,  1.9817e-02,
 1.1420e-01, -5.1386e-03,  6.8617e-03,
-1.3264e-02,  2.1508e-01,  4.8430e-02,
 5.1149e-02,  2.9165e-01,  2.8077e-01,
 2.9288e-03,  9.0611e-02,  8.1538e-02,
-1.1812e-01,  1.5603e-02,  1.1571e-01,
-3.4958e-02, -1.6688e-03, -4.6619e-02,
-1.0417e-02, -3.1802e-02,  1.8357e-02,
 1.1064e-01,  1.8397e-01,  4.8449e-02,
-8.3336e-03,  1.6029e-01,  3.9490e-02,
-4.0959e-01, -2.6134e-01,  2.0766e-02,
 6.6073e-41, -6.7490e-40, -5.1131e-41,
-4.3320e-41, -3.7194e-40,  2.0674e-40,
-5.2359e-40, -3.4006e-40, -4.9257e-40,
-4.7260e-02,  2.8518e-03, -2.7764e-01,
 6.9182e-03,  1.3938e-01, -1.3162e-01,
-6.0901e-03,  1.0339e-01,  6.0419e-02,
-1.4449e-01, -3.2043e-02, -9.1466e-02,
-1.4022e-02,  3.1703e-01,  5.8166e-02,
-1.5243e-02,  1.4521e-01,  2.0790e-04,
-1.0255e-01, -7.8766e-02, -1.2395e-01,
 7.9894e-03,  3.7079e-03, -3.2134e-02,
 1.1663e-01,  1.4808e-01,  2.0431e-01,
 7.4026e-02,  6.9632e-02,  1.7156e-01,
-3.0385e-02,  2.3218e-01,  7.3855e-02,
-8.8530e-02, -5.9224e-02,  2.3431e-02,
 1.4596e-02,  3.2442e-02, -1.1308e-01,
-6.3734e-02,  2.5270e-01,  7.8081e-02,
 1.0468e-02,  1.5473e-01,  3.8676e-02,
-1.0842e-01,  8.6778e-03,  1.4985e-01,
 8.1757e-03, -8.2109e-02,  8.5471e-02,
-2.1437e-01, -6.1173e-02,  4.8163e-02,
 2.8965e-01,  1.9748e-01,  4.2651e-02,
 1.8196e-01,  3.3932e-01,  3.9594e-01,
 3.9657e-01,  4.2167e-01,  2.9290e-01,
 7.4011e-41,  6.5220e-40, -5.9885e-40,
 7.4011e-41,  6.2047e-40, -7.1533e-40,
 4.1950e-40, -1.1886e-40, -5.9922e-40,
 1.9662e-01,  2.1402e-01,  3.1041e-02,
-1.1079e-01,  1.3361e-01, -2.1608e-01,
-1.7962e-01, -8.0576e-02, -3.1277e-01,
 1.0620e-02,  2.4024e-01,  1.0657e-01,
-7.9906e-05,  2.8760e-01,  4.1231e-02,
-1.3261e-02, -1.0868e-01, -1.1267e-01,
-1.0659e-02, -2.6051e-02, -4.5389e-02,
 5.8261e-02,  4.0288e-02,  6.7050e-02,
-2.6462e-01, -1.7846e-01, -1.0002e-01,
-6.2904e-02,  1.5275e-01,  4.4282e-03,
 1.4446e-01,  1.1814e-01, -8.0349e-02,
 2.0331e-02,  3.3014e-02,  1.2710e-01,
 1.6084e-01,  3.8819e-01,  1.0854e-01,
-6.8126e-03,  3.5673e-01,  1.8938e-01,
-1.1660e-01, -5.7694e-02, -2.9194e-01,
 1.2775e-02, -3.2769e-02,  1.7228e-02,
 1.8324e-01,  1.1983e-01, -1.6944e-02,
 1.0593e-01,  1.3451e-01,  5.2536e-02,
 1.9147e-01,  1.3875e-01,  1.0298e-01,
-2.0871e-01, -1.7197e-01,  1.1342e-01,
-1.7581e-01,  4.0972e-02,  2.9796e-01,
 3.2588e-40, -4.3663e-40, -2.6518e-40,
 3.2588e-40, -4.3663e-40, -2.6518e-40,
 4.1600e-40, -4.4350e-40, -4.8744e-41,
 3.7289e-02,  8.1769e-03,  1.7059e-02,
 3.7735e-02,  6.6571e-02, -6.6137e-02,
-5.8890e-02, -7.7019e-03, -6.2128e-02,
-4.0751e-02,  1.1710e-01, -1.1586e-01,
-1.2999e-01, -1.6384e-02, -2.1858e-01,
-2.8028e-01, -6.0443e-02, -1.1880e-01,
 1.8152e-01,  1.5364e-01,  1.1781e-01,
 2.9010e-01,  2.4612e-01,  1.3170e-01,
 1.9022e-01,  1.8117e-01,  1.6483e-01,
 9.3342e-02,  2.6607e-01,  1.4679e-01,
 1.6729e-01,  2.5374e-01,  1.1954e-01,
 6.3258e-02,  1.0557e-01,  6.7221e-02,
-5.2017e-02,  1.9628e-01,  1.7243e-01,
-3.2667e-02,  1.5756e-01,  1.9347e-01,
-9.5252e-02, -3.7525e-02, -3.4543e-04,
-4.9759e-02,  4.0383e-02, -2.0231e-02,
-1.1776e-01,  3.4182e-02,  3.6720e-02,
-1.4822e-02, -4.1658e-02, -1.3729e-02,
-1.9215e-02,  2.4427e-02, -9.0638e-02,
-1.4438e-01, -2.1785e-01, -5.1789e-02,
-2.0279e-01, -3.3918e-01, -1.6871e-01,
 6.1262e-41,  2.4066e-40,  6.6851e-40,
 5.3430e-40, -3.2335e-40, -3.7400e-40,
-6.3256e-40, -4.7491e-40,  2.2854e-40,
-6.8701e-03, -1.4849e-02,  8.6332e-02,
 1.1686e-01,  1.8346e-01,  1.8797e-01,
-2.3251e-02,  7.3973e-02,  1.0532e-01,
-6.1838e-02,  5.6667e-02,  8.1584e-02,
-3.8900e-02,  7.0927e-02,  9.5606e-02,
-4.5098e-02, -1.0829e-01, -1.2224e-01,
 3.5047e-03,  3.2898e-02,  3.5622e-02,
 1.6170e-02,  4.3721e-02,  9.7496e-02,
 2.3445e-03,  6.0417e-02,  1.3482e-01,
 6.0570e-02, -5.7139e-03, -1.0883e-03,
 2.2701e-02, -2.9113e-02,  7.9178e-03,
 8.1214e-02, -4.1408e-02,  1.3616e-02,
-4.7985e-02,  1.0304e-02, -3.3236e-02,
-1.6334e-02, -8.1538e-02,  1.8629e-02,
-9.3720e-02, -1.2920e-01, -4.0836e-02
}
,)"
R"(
{
 1.0443e-01,  1.5461e-01, -1.4743e-01,
 1.6716e-01,  1.0532e-01, -2.3088e-01,
 1.0218e-01,  1.2393e-01, -9.6646e-02,
 1.7659e-01, -7.3279e-02,  1.9627e-02,
 1.7721e-01, -1.4329e-01, -1.2533e-01,
 1.6551e-01, -3.4616e-01,  9.5618e-02,
 4.5827e-09,  9.3413e-09,  1.7015e-08,
 1.2245e-08,  9.9727e-09,  6.7108e-09,
 1.9612e-07,  3.9479e-08,  1.1537e-09,
 2.2127e-02,  9.2715e-02, -1.2150e-01,
 7.5652e-02,  1.1548e-01, -1.2420e-01,
-1.0693e-03, -7.2839e-02, -1.9664e-01,
 1.4466e-01, -1.8552e-03, -1.3575e-01,
 2.0699e-01,  8.0396e-02, -1.9651e-01,
-4.7075e-02, -5.1259e-02, -8.2593e-02,
-2.2385e-01,  3.0066e-03, -2.2659e-02,
 6.1827e-02,  2.5331e-02, -5.3898e-02,
 2.7091e-01,  1.0991e-01, -3.3600e-01,
-8.9499e-02, -9.3821e-03,  2.2675e-02,
 1.1213e-01,  1.3276e-01,  2.0368e-02,
 6.5408e-02,  4.1598e-02, -4.7917e-02,
 6.0740e-03,  1.2236e-04, -1.0659e-01,
-1.8072e-02, -9.1082e-02, -9.0414e-02,
 4.9052e-02, -1.4298e-01, -3.9721e-02,
 1.1840e-01,  2.2503e-01,  2.4587e-02,
 9.3023e-02,  6.9650e-02,  1.6798e-01,
-1.5640e-03,  1.6300e-02,  6.3585e-02,
 1.4431e-01,  3.7885e-02,  1.6692e-02,
 1.7345e-01,  7.2315e-02,  1.8942e-02,
 1.1081e-01,  8.2973e-02, -9.7717e-02,
-5.2264e-03, -5.2641e-03, -5.2727e-03,
-5.2809e-03, -5.3125e-03, -5.3153e-03,
-5.2915e-03, -5.3251e-03, -5.3231e-03,
 6.0008e-02,  2.0268e-01,  1.3396e-01,
-2.5202e-03, -1.7750e-02, -1.2019e-02,
 1.1806e-01, -2.2306e-02,  3.6464e-02,
 7.9324e-02,  3.1883e-02,  1.5483e-02,
-4.3537e-02,  1.2204e-02,  1.8905e-02,
-8.1581e-02, -1.1307e-01, -6.0718e-02,
-2.4865e-01, -1.0199e-01,  1.9886e-02,
-1.0519e-02,  6.9972e-02,  4.8012e-02,
-1.5282e-02,  1.1979e-01,  8.7968e-02,
-3.6752e-02,  1.9523e-02,  7.1321e-02,
-5.8295e-02,  5.3242e-02,  1.2773e-01,
-7.9671e-02,  8.3249e-04,  7.4904e-02,
 1.1792e-01,  2.2135e-03, -9.0963e-03,
-2.8356e-03, -4.2661e-02,  6.9497e-02,
 9.3561e-02,  1.0475e-01,  5.4745e-02,
-8.5901e-02, -2.1969e-01, -1.5572e-01,
 3.6473e-02,  1.1097e-01, -2.6830e-02,
 1.2199e-02,  1.8917e-01,  1.1906e-01,
 1.0664e-01, -2.7005e-01,  1.5492e-01,
-4.1771e-02, -1.6580e-01,  2.9234e-02,
-1.9854e-02,  2.1436e-01, -1.1100e-01,
 4.5382e-04,  4.2085e-04,  5.6852e-04,
 3.4951e-04,  3.7354e-04,  3.2786e-04,
 2.0790e-04,  2.8606e-04,  3.2415e-04,
-1.5500e-02,  2.2865e-02, -3.0070e-01,
 1.8467e-01,  2.4899e-01,  1.4812e-02,
-1.2318e-01,  2.3175e-01,  7.2244e-02,
 1.6713e-01,  1.9089e-02, -2.7494e-01,
 1.0202e-01,  2.9200e-01, -3.6055e-03,
 1.3265e-01,  2.2551e-01,  1.9897e-01,
-3.9474e-02,  1.6262e-01,  1.6726e-01,
-8.6222e-02,  2.0573e-01, -7.3247e-01,
-9.5391e-02,  3.8933e-01,  1.5861e-01,
-1.2202e-01, -6.4735e-02, -1.1762e-01,
-2.2427e-02, -1.9171e-01, -1.6092e-01,
 3.2356e-01, -2.2234e-01, -1.3743e-01,
-1.1493e-01, -2.4936e-02,  2.9212e-02,
-9.8112e-02, -1.8021e-02, -1.0507e-01,
-1.0168e-01,  1.1759e-01, -9.8203e-02,
-2.8871e-02,  1.3249e-01,  7.8378e-02,
-1.1012e-01, -4.0596e-02,  5.4202e-02,
 4.9022e-02, -1.1744e-01,  9.8888e-02,
 1.3343e-02,  1.4358e-01, -8.7142e-02,
 1.9952e-01,  3.3708e-02,  2.0721e-02,
 2.6527e-02, -2.3822e-01,  2.4706e-01,
-3.2750e-04, -2.8475e-04, -6.3494e-05,
-2.2378e-04, -1.8046e-04, -1.9242e-05,
-4.2124e-05, -2.2062e-05,  4.5500e-07,
 1.1692e-01,  4.0366e-01, -1.8709e-02,
 8.2700e-02,  1.7884e-01, -1.3520e-01,
 3.7758e-02,  3.7048e-02, -2.8109e-01,
-2.3438e-01,  5.9423e-02, -1.7300e-01,
 1.0343e-02,  7.2307e-02, -4.3852e-01,
-5.7429e-02, -4.9136e-02, -8.0327e-02,
 8.1094e-02,  2.9118e-02,  1.6677e-01,
 1.2155e-01,  6.5358e-01,  2.4544e-01,
 3.1163e-02,  3.7463e-02, -2.6613e-01,
 1.2723e-01,  1.2541e-01,  1.4319e-02,
 1.9055e-01, -5.7441e-02,  1.1146e-01,
-1.0690e-02, -1.7567e-01, -1.2238e-01,
-2.0879e-01, -6.5278e-02, -7.9327e-02,
-1.6564e-01, -1.3659e-01, -2.6231e-01,
-3.1916e-01, -2.6553e-01, -9.8647e-02,
-1.0617e-01,  1.2782e-01, -2.1053e-02,
-1.2329e-01,  1.4952e-01, -1.7466e-02,
-1.6969e-01,  3.6980e-02, -6.7732e-02,
-3.1220e-02,  4.0615e-02, -1.5251e-01,
-2.0017e-01,  2.2421e-01, -2.5682e-02,
-6.5873e-02,  1.8346e-01,  1.2982e-02,
 1.4021e-06, -1.6929e-05, -8.4696e-05,
 1.9580e-05,  2.9943e-06,  3.0084e-06,
 2.0769e-04,  1.4661e-05,  2.9503e-06,
-1.4485e-01,  1.8841e-01, -1.7954e-01,
 2.1551e-01,  2.2601e-01, -8.6689e-03,
 8.6926e-02, -6.8989e-02, -1.2683e-01,
-8.7712e-02,  6.3176e-02,  1.1983e-01,
 1.0790e-01,  6.6418e-02,  6.5849e-02,
 1.2483e-01,  1.2428e-01,  4.4994e-02,
 1.5139e-01, -1.2116e-01, -3.5497e-01,
-6.1889e-02,  3.4088e-01,  1.3148e-01,
-1.6478e-01,  4.4477e-02, -1.1979e-01,
 3.8343e-02,  1.7992e-01,  3.6790e-01,
 3.0426e-01,  1.1235e-01,  4.9815e-01,
 2.6290e-01,  1.9703e-01,  1.5881e-01,
-6.4678e-03,  2.4401e-01,  1.9266e-01,
-1.4089e-01,  1.2323e-01,  4.4340e-02,
-8.8856e-02,  8.4036e-02, -9.8488e-02,
-1.7377e-03, -1.7654e-03, -1.7223e-03,
-1.7651e-03, -1.7919e-03, -1.7491e-03,
-1.7172e-03, -1.7446e-03, -1.7041e-03,
-3.0384e-04, -2.9297e-04, -2.4838e-04,
-3.2961e-04, -3.1678e-04, -2.7009e-04,
-3.1665e-04, -3.0492e-04, -2.6122e-04,
 3.7109e-40, -3.7915e-40, -5.2536e-40,
 5.8286e-41, -5.6108e-40,  4.3331e-40,
-3.0184e-42, -4.8987e-40, -5.1788e-40,
-4.0457e-04, -4.3257e-04, -4.1616e-04,
-4.2268e-04, -4.5118e-04, -4.3407e-04,
-3.9446e-04, -4.2199e-04, -4.0650e-04,
-1.1253e-16, -1.1328e-14, -2.0489e-14,
-3.0346e-19, -1.7189e-16, -4.5141e-16,
-2.4957e-30, -1.8191e-23, -3.5882e-22,
-3.1610e-36, -1.7544e-24, -2.2187e-21,
-4.2887e-19, -1.5526e-15, -1.5160e-14,
-1.7750e-16, -6.8066e-14, -3.3764e-13,
-6.9570e-24, -5.1139e-23, -2.9335e-23,
-1.9091e-22, -1.0323e-21, -4.5931e-22,
-2.0010e-22, -9.3710e-22, -3.5622e-22,
-2.9470e-04, -2.9081e-04, -2.5958e-04,
-3.2290e-04, -3.1810e-04, -2.8461e-04,
-3.1795e-04, -3.1356e-04, -2.8121e-04,
 6.1623e-02,  1.7057e-01,  8.0478e-02,
 1.2624e-01,  1.8468e-01,  2.1901e-02,
 7.6033e-02,  1.3455e-01,  8.4037e-02,
 8.4434e-02, -1.7069e-02, -7.8318e-02,
 4.9244e-02,  4.4782e-02, -6.9747e-02,
 1.2915e-01,  1.1453e-01, -6.5243e-02,
-5.0985e-03, -5.1407e-03, -5.1687e-03,
-5.1185e-03, -5.1511e-03, -5.1712e-03,
-5.0986e-03, -5.1272e-03, -5.1409e-03,
-1.8186e-02,  6.2680e-02,  3.3235e-02,
 1.3398e-02,  1.6497e-01,  4.3523e-02,
-2.4101e-02,  1.3316e-01,  1.8373e-02,
-6.2677e-04,  6.5026e-03,  2.5948e-02,
 6.6542e-02,  1.2352e-01,  1.5155e-02,
-8.6237e-02, -2.0907e-02,  1.0237e-02,
-1.7807e-01, -8.6196e-02, -3.2408e-02,
-8.1946e-03, -1.3957e-02, -1.6733e-01,
 2.6269e-02,  1.6817e-01,  9.4029e-02,
 3.4005e-02, -1.2833e-02, -1.2038e-01,
-4.8950e-02,  3.9857e-02,  1.4048e-02,
-6.4758e-02,  9.9603e-02,  1.0748e-01,
-1.0850e-02,  9.8875e-02, -4.4439e-02,
 9.1219e-02,  6.6400e-02, -6.7693e-02,
 5.3318e-02,  1.1838e-02, -1.5164e-01,
-5.8568e-02,  1.1249e-01, -3.8286e-02,
-7.1122e-02,  9.5799e-02,  3.8521e-02,
-1.3846e-01,  1.4167e-01, -3.5500e-03,
-1.0343e-01, -3.3025e-02,  3.7186e-02,
-2.0769e-03,  1.3558e-01, -1.3009e-01,
 1.0167e-02,  1.5358e-02, -9.8009e-02,
 2.4123e-05, -1.1800e-05, -1.4180e-04,
 3.5217e-05, -6.3838e-06, -1.2243e-04,
 8.5525e-05,  2.1599e-06, -5.3290e-05,
-1.4471e-01,  2.0111e-02, -1.2449e-01,
 5.3368e-02,  3.2918e-01,  1.4034e-01,
-1.1833e-01, -1.9225e-02, -1.2658e-01,
-2.6966e-01,  1.1751e-01,  9.7072e-02,
-1.9929e-01,  9.7986e-02, -5.1240e-02,
-9.5073e-02, -6.8070e-02, -2.1318e-01,
 9.5305e-02, -4.0551e-02, -1.0936e-01,
 5.2687e-02,  4.5340e-01,  2.3531e-01,
-1.3385e-02,  1.5922e-01, -1.8371e-01,
-1.2203e-01, -7.2567e-02, -3.0000e-01,
-3.4356e-02, -1.3471e-01, -9.0995e-02,
-2.5230e-01, -2.4846e-01, -1.8529e-01,
-1.6962e-01,  1.0905e-01,  1.1557e-01,
-1.4405e-01,  8.9191e-02,  1.1715e-01,
-1.3237e-01,  5.2092e-02, -1.2227e-01
}
,)"
R"(
{
 2.0013e-01,  2.2105e-01,  1.9196e-01,
 6.8158e-02,  1.7154e-01, -8.6677e-02,
 9.2652e-02,  1.0789e-01,  1.6745e-01,
-2.9254e-01, -7.6815e-02,  5.8812e-02,
-4.6466e-02,  1.3941e-02,  2.3353e-01,
-1.5033e-01,  7.5167e-02,  1.4433e-01,
 2.8008e-02,  3.1625e-01,  3.2877e-02,
-5.8835e-02, -1.7305e-01, -6.1558e-02,
-1.2227e-01,  3.9931e-02,  3.0300e-02,
 2.3004e-01,  4.1834e-02, -5.7790e-02,
-2.2861e-01,  2.9314e-01,  1.6884e-01,
-2.8009e-02,  4.7550e-02, -4.4542e-02,
-2.4674e-01, -1.5483e-01,  3.2653e-02,
-2.1574e-01,  3.1083e-01, -1.4025e-03,
 1.7354e-02,  5.6417e-02,  1.0844e-01,
-4.2681e-40,  4.5893e-42, -7.4234e-40,
 1.7665e-40,  4.0151e-40,  4.6269e-40,
 2.5452e-40, -7.0179e-40, -1.2338e-40,
-1.4957e-01, -1.9087e-02,  7.1170e-02,
-1.4435e-01,  8.9560e-02,  1.3879e-01,
-3.6992e-02,  5.9822e-02,  1.9241e-02,
-2.4402e-03,  1.5097e-01,  6.3958e-02,
-1.7630e-01,  3.6009e-01, -2.0383e-01,
-8.5106e-03,  4.0863e-03, -2.7575e-02,
 7.8942e-02, -1.8640e-01, -6.7715e-02,
 7.2777e-02, -1.3804e-01, -7.0332e-02,
 1.5185e-01, -4.3530e-02,  1.4502e-01,
-3.2928e-02, -3.0583e-02,  9.2061e-02,
 1.2493e-01,  1.0400e-01,  1.3780e-01,
 1.4438e-01,  8.2051e-02,  1.6159e-02,
 2.7478e-02,  1.7768e-01,  2.5945e-01,
-3.4662e-01,  2.0330e-03,  8.8118e-02,
-2.9628e-01, -1.3212e-01, -1.8145e-02,
-1.9330e-01,  3.9238e-02, -4.6944e-02,
-1.5668e-01, -5.7104e-02,  1.9558e-01,
 6.5305e-02,  5.9933e-02,  7.7337e-02,
-2.4906e-02, -1.1235e-01,  1.3822e-02,
-3.9988e-02, -9.1882e-03,  1.9204e-02,
 1.0504e-01,  4.6820e-03, -2.1836e-02,
-2.6953e-40,  2.5334e-40, -1.3028e-40,
 1.4110e-41,  5.6841e-40,  3.6368e-40,
-1.1746e-41, -7.0658e-41, -3.9413e-40,
 1.5025e-02,  7.4419e-02,  9.5652e-02,
 5.0297e-02,  6.6704e-02,  5.7316e-02,
 2.5102e-02,  1.1985e-01,  2.6043e-02,
 3.3297e-02, -7.7374e-02, -1.1114e-01,
-7.5586e-02, -1.9338e-02, -1.3739e-02,
 4.5616e-02, -6.4946e-02, -6.9372e-02,
-7.5874e-03, -1.1141e-01, -2.9135e-02,
-6.9436e-03, -1.4418e-02,  1.6436e-03,
-1.3051e-01, -1.3324e-01, -9.3934e-02,
 1.2184e-01,  1.9386e-01,  1.7995e-01,
-2.7452e-02,  9.9736e-02,  1.0020e-01,
-6.3290e-02, -2.1447e-02, -1.7005e-01,
 1.3857e-01,  2.3338e-01,  2.5410e-01,
 2.3002e-01,  1.9551e-01,  1.4452e-01,
 4.7040e-01,  2.2647e-01,  1.5215e-01,
 2.6927e-02, -2.1304e-01, -1.4762e-01,
-5.6998e-02,  2.9064e-01,  1.8085e-01,
 8.9393e-02, -1.7463e-01, -2.7095e-01,
 3.8434e-02,  1.7198e-01, -1.8122e-02,
-1.3857e-01,  1.9418e-01,  1.5019e-01,
-5.6337e-02, -5.3265e-01,  3.2122e-01,
-2.4484e-40, -5.3707e-40,  1.5854e-41,
 5.1791e-40, -4.1875e-41,  5.6732e-40,
 1.3048e-40,  1.6452e-40, -4.5028e-40,
-3.0692e-02,  1.8569e-01,  2.0327e-01,
-7.4756e-02, -5.1765e-02,  4.2475e-02,
-9.0675e-02, -3.0438e-01, -3.5088e-01,
-1.9129e-02, -1.5663e-03,  4.9895e-02,
-1.9441e-02,  9.3237e-02,  1.2910e-01,
-2.3919e-02, -4.0539e-01,  2.8167e-02,
 2.0203e-01,  3.3424e-02,  1.7927e-02,
 4.1923e-02, -1.6967e-01,  2.5656e-02,
-1.5869e-01, -1.8727e-01,  2.7860e-03,
-4.0276e-02, -6.7792e-03,  3.3699e-02,
-6.7044e-03,  1.7686e-02,  2.9786e-02,
-1.5623e-02,  3.7904e-02,  2.4737e-02,
-1.2282e-01, -3.6563e-02,  4.1976e-02,
-9.9622e-03,  8.8981e-02,  2.1364e-02,
-8.5668e-02, -1.6803e-01, -4.4974e-02,
 1.3164e-01,  4.1294e-01,  1.8897e-01,
 2.1991e-01,  1.6247e-02,  1.1569e-01,
-3.0142e-02,  1.4069e-02,  3.6646e-02,
-2.6816e-02, -3.9767e-02,  1.4061e-01,
-1.3603e-01, -2.0649e-01,  7.5837e-02,
-1.6984e-02, -8.3800e-03,  2.3652e-04,
 1.5049e-40,  4.6504e-40,  1.3625e-40,
-7.5358e-40, -3.4257e-40,  9.9763e-41,
 4.7243e-40,  7.4890e-40, -7.9440e-42,
-5.9692e-02, -2.8047e-02,  2.3795e-02,
-3.5284e-02,  1.1448e-02,  5.0302e-04,
-3.5066e-02,  4.6185e-02,  1.2167e-02,
 3.7583e-02, -3.6598e-02,  1.0206e-01,
-9.6229e-02, -1.5977e-01,  4.9157e-02,
 3.7293e-02,  5.8766e-02,  1.0448e-02,
 1.1490e-01,  1.4459e-01,  8.6936e-02,
 2.8609e-01, -4.8108e-02,  9.0023e-02,
 6.7941e-02, -5.7148e-03,  1.0021e-01,
 7.3816e-02,  7.3794e-02,  8.0970e-03,
 2.8307e-02,  3.6635e-03, -1.1769e-01,
 4.1374e-02,  3.9933e-02, -4.4292e-02,
 5.9423e-02,  1.9009e-01, -2.3735e-01,
-2.6670e-01,  5.8789e-01, -2.0048e-01,
-3.7082e-01,  1.8045e-01,  5.4820e-02,
-6.3567e-01,  2.0098e-01,  1.0653e-01,
-2.5056e-01,  6.5065e-01, -4.0471e-01,
 5.4715e-02,  2.4375e-01, -2.7402e-01,
 1.5982e-01,  1.0923e-01,  2.1566e-01,
 2.0239e-01, -9.0221e-02, -4.4606e-01,
 1.0550e-01,  5.4666e-02, -2.7134e-01,
-4.6424e-40,  2.9137e-40,  7.4968e-41,
 1.2376e-41, -5.6213e-40, -6.3457e-40,
 2.5404e-40,  2.0013e-40,  3.5611e-40,
 5.5423e-02,  3.9843e-02, -1.7509e-01,
 5.4480e-02,  5.0331e-02, -1.6793e-01,
 6.6093e-02,  3.0163e-02, -8.2023e-02,
-1.5490e-01,  1.7457e-01,  2.7832e-01,
 1.1482e-01,  2.5759e-01, -2.4199e-01,
-9.3891e-02,  9.1921e-02, -6.4480e-03,
 1.9266e-01,  5.2907e-02,  7.0289e-02,
 1.3582e-01,  6.4246e-02,  1.4989e-01,
 6.2013e-03, -6.8884e-02,  6.8734e-02,
-1.0483e-01, -7.7134e-02, -3.6204e-02,
 1.7590e-02,  5.0844e-02,  1.4234e-01,
 7.2913e-02,  6.0726e-02,  6.4414e-02,
-8.5021e-02, -1.0621e-03,  5.5851e-02,
 2.4666e-01,  6.5652e-02, -1.8180e-02,
 1.5225e-01,  1.2928e-01,  3.1578e-03,
 1.1468e-01,  1.9544e-01,  6.6637e-02,
 6.3430e-02,  2.0542e-01,  7.0876e-02,
 3.4779e-02,  1.0037e-02, -2.2134e-02,
-6.9304e-02,  1.1184e-01, -3.7015e-02,
-1.7634e-01,  1.2475e-01,  9.1947e-02,
-6.0550e-02, -1.3904e-01,  7.5192e-02,
-2.2871e-40,  4.7367e-41, -1.0711e-40,
-2.8662e-40,  4.0542e-41,  3.3067e-40,
-4.4395e-41, -7.2684e-41,  1.8695e-40,
-1.6702e-01, -2.6654e-01,  8.7902e-03,
-2.0108e-01, -3.8093e-01, -8.3700e-02,
-7.5433e-02, -2.0689e-01,  2.7951e-02,
 2.9938e-03,  1.1378e-01,  7.1598e-02,
-1.6031e-01,  1.3475e-01,  1.5800e-01,
-7.2019e-02, -1.1663e-01,  8.0692e-02,
 1.0610e-01,  1.1163e-02, -1.4959e-01,
-1.1576e-01, -8.5645e-02,  4.0414e-02,
 5.6245e-02,  1.7056e-01,  2.5734e-01,
-6.1086e-02, -7.0851e-02,  7.6851e-02,
-2.7595e-02, -6.0890e-02,  4.7472e-02,
 7.1059e-03,  6.0942e-05,  7.4915e-02,
 1.9350e-01, -1.8458e-02, -2.3040e-02,
 6.3477e-02,  1.1923e-01,  9.9319e-02,
 6.4839e-02,  2.7973e-01,  1.2902e-01,
-1.7829e-01,  5.7083e-03, -6.1680e-03,
-1.1256e-01, -2.7951e-02, -2.1544e-01,
-2.1614e-02, -7.1468e-02, -2.2054e-02,
-8.7543e-02, -1.2982e-01,  1.9386e-01,
-5.7157e-03, -1.0108e-01,  1.4467e-01,
-6.5742e-02, -7.2054e-02,  1.7924e-01,
 7.5418e-40,  6.3043e-40,  4.9815e-40,
-1.0952e-40,  3.0327e-40, -2.3848e-40,
 4.1302e-40,  2.0150e-40, -1.6509e-40,
-1.3985e-02, -1.0550e-01,  5.8772e-02,
-1.7108e-02, -7.3644e-02,  3.3014e-02,
-1.8224e-03,  2.8931e-03,  9.2762e-02,
 4.1531e-02, -1.5139e-01, -1.7773e-01,
 9.6548e-02, -1.1914e-01, -4.6536e-02,
 8.6754e-02, -4.0057e-03,  1.8983e-01,
 1.6545e-01, -4.7311e-02, -7.2455e-03,
 3.7567e-01,  1.8883e-01, -7.4325e-02,
-5.8252e-02, -1.3811e-02, -7.0470e-02,
-3.2943e-02, -7.0770e-02, -1.4700e-01,
 1.7043e-02,  9.4331e-02,  4.2857e-03,
 4.1247e-03,  1.6690e-01,  4.2146e-02,
 1.1420e-01, -7.4456e-02, -3.8763e-02,
 1.6807e-01,  9.3636e-03, -1.1796e-01,
 1.7703e-01,  1.1386e-03, -6.8707e-02,
 1.0259e-01, -1.8918e-02,  6.5902e-03,
 1.2421e-02, -7.8960e-02,  2.1766e-02,
 1.3062e-01,  4.6001e-02,  2.4199e-01,
-1.2955e-02, -1.9329e-01,  5.2074e-03,
 5.9446e-02,  1.8832e-01,  2.2094e-01,
-1.0954e-01, -8.1867e-02, -4.3324e-02,
-3.9596e-41,  2.8677e-40, -6.5843e-40,
 4.2812e-41, -3.5323e-40,  4.8298e-40,
 7.6351e-40, -2.4759e-40,  7.3030e-40,
-1.1284e-01, -8.4171e-02, -1.5935e-01,
-3.2299e-02,  1.5427e-01,  8.9029e-02,
-3.8815e-02,  1.3098e-01, -4.3065e-02,
-2.5276e-01, -1.7018e-01,  9.7901e-02,
 1.4218e-01,  3.1236e-01,  2.9636e-01,
-2.3613e-02, -5.5258e-02, -2.0550e-01
}
,)"
R"(
{
 0.0333,  0.1145, -0.0922,
 0.1185,  0.4533, -0.2015,
-0.0774,  0.1759, -0.0496,
 0.0954, -0.0499,  0.0824,
 0.1059,  0.0173, -0.0586,
-0.0666, -0.0287, -0.0652,
-0.0558, -0.1362,  0.0015,
 0.1277,  0.1020, -0.1369,
 0.0020, -0.0103, -0.0804,
 0.0507,  0.1404, -0.0241,
 0.0520,  0.1239,  0.0633,
-0.0268,  0.0335,  0.0883,
-0.0549, -0.1022, -0.0515,
-0.0163, -0.1167, -0.0442,
 0.0858, -0.0804, -0.0014,
 0.0354, -0.0666, -0.2105,
-0.0950,  0.1578, -0.0920,
-0.1303,  0.0299, -0.0195,
-0.0281, -0.1993, -0.0154,
 0.0796,  0.0503,  0.0954,
 0.0540,  0.0212,  0.0389,
-0.1387,  0.1091, -0.1212,
 0.1556,  0.3573,  0.0976,
-0.0587, -0.2070,  0.2067,
 0.0138,  0.0051, -0.1008,
 0.2877,  0.1079, -0.0681,
 0.0953, -0.0739, -0.2349,
 0.1482,  0.0657,  0.0480,
 0.1590, -0.0009,  0.1402,
 0.0700,  0.0435,  0.1190,
 0.0957,  0.0117, -0.1010,
 0.1790, -0.0200, -0.0765,
 0.0797,  0.1455, -0.0340,
 0.0008, -0.0267,  0.0089,
 0.0644,  0.0647,  0.0397,
 0.0463, -0.0116, -0.0771,
 0.2237,  0.0324,  0.0192,
-0.0082, -0.0345,  0.0294,
 0.0719, -0.0185,  0.1008,
-0.0307,  0.0134, -0.0747,
 0.0776, -0.1485,  0.0135,
 0.0965, -0.0665, -0.1263,
-0.0101, -0.0097, -0.0144,
-0.0022, -0.0083,  0.0277,
 0.0136, -0.0076,  0.0314,
-0.0008,  0.0722, -0.0704,
 0.0053,  0.0767,  0.0368,
-0.0189, -0.1354,  0.0231,
-0.1416,  0.1945, -0.1756,
 0.2058,  0.0401, -0.1348,
-0.0945, -0.2530, -0.3082,
-0.0096,  0.0871,  0.0699,
-0.0092,  0.0423,  0.0995,
-0.0914, -0.0570, -0.0718,
-0.0739, -0.2749, -0.2320,
 0.1488, -0.2698, -0.1977,
 0.1445, -0.1655, -0.0758,
 0.2035, -0.0138,  0.0332,
 0.0282, -0.2247, -0.0945,
-0.0614, -0.2484, -0.0595,
-0.1174, -0.1252,  0.1969,
-0.1101, -0.2950, -0.2164,
-0.0348, -0.0891,  0.1250,
 0.0195,  0.0050,  0.0300,
-0.0508, -0.0316, -0.0194,
 0.0199,  0.0345,  0.0444,
-0.0022, -0.0529,  0.1604,
 0.0756, -0.2015, -0.2117,
-0.0837, -0.1270,  0.1330,
 0.0286,  0.0952,  0.1082,
 0.0724, -0.0446, -0.1156,
 0.0545,  0.0444, -0.0291,
 0.0759,  0.1110,  0.0944,
 0.1615,  0.4302, -0.1060,
 0.0418, -0.0281, -0.1378,
-0.0757, -0.0527, -0.1578,
 0.0123, -0.0427,  0.1504,
 0.0694,  0.0690,  0.0203,
 0.2132, -0.3449,  0.0936,
 0.2491,  0.0279, -0.0884,
-0.0447,  0.1589, -0.0054,
-0.0246,  0.1247,  0.0403,
 0.0513, -0.0541, -0.1141,
 0.0712, -0.1174, -0.0051,
 0.2304,  0.2431, -0.0517,
-0.1548, -0.0401,  0.2032,
-0.0087, -0.1676, -0.0600,
 0.1094, -0.0329,  0.0530,
-0.0580,  0.1499, -0.0806,
-0.0086, -0.1400, -0.0636,
 0.0708, -0.1003, -0.1113,
-0.0732, -0.1199,  0.0060,
-0.0534, -0.0011,  0.0965,
-0.0268,  0.0116, -0.1161,
 0.0787,  0.3925, -0.0819,
-0.0041, -0.0892, -0.2063,
-0.1296,  0.0924, -0.0079,
 0.5625,  0.4013,  0.1645,
-0.0137, -0.1935,  0.2714,
 0.0980,  0.0016, -0.1461,
 0.1576,  0.0305, -0.1450,
 0.1503, -0.0303, -0.1403,
 0.0262, -0.0077,  0.0459,
 0.2718,  0.0754,  0.2404,
 0.1381, -0.1499,  0.0016,
 0.1454, -0.1278, -0.0085,
 0.1674, -0.0834,  0.1993,
 0.0874, -0.0598, -0.0188,
 0.2003,  0.3296,  0.0153,
-0.0154,  0.5550, -0.0945,
 0.0489,  0.0415, -0.0940,
 0.0164,  0.0791,  0.1077,
-0.0893,  0.1231,  0.0473,
-0.0319,  0.1444,  0.1690,
-0.0518, -0.1404, -0.1778,
-0.0170,  0.1395, -0.0234,
 0.0128, -0.0112, -0.0472,
 0.1039,  0.1982, -0.0272,
 0.0282, -0.1199, -0.2622,
-0.0449,  0.0239, -0.1030,
-0.0840, -0.1044, -0.0646,
 0.0588,  0.1937, -0.2494,
 0.0180,  0.0747,  0.1530,
 0.0500,  0.1756,  0.0491,
-0.1113, -0.0079,  0.0854,
-0.1493, -0.0559, -0.0373,
 0.1972, -0.3158, -0.0500,
 0.1932,  0.3177, -0.0018,
-0.0516, -0.1144,  0.0686,
 0.0175,  0.0598,  0.0345,
-0.0667, -0.1078,  0.0384,
 0.0897,  0.2198, -0.0531,
-0.2596, -0.1997,  0.0195,
 0.0332,  0.4098,  0.1381,
 0.1985, -0.0669, -0.1275,
-0.0751, -0.2388, -0.0672,
 0.0090,  0.0891, -0.0362,
 0.1392, -0.0518,  0.2039,
 0.2079, -0.1202,  0.0707,
 0.0498, -0.1237, -0.0665,
-0.0398, -0.1557, -0.0928,
 0.0505,  0.1220,  0.0352,
-0.0674, -0.1159,  0.0724,
-0.0331, -0.1751,  0.0766,
 0.0992, -0.0763,  0.0090,
-0.1223,  0.2621, -0.2029,
 0.0509, -0.0279, -0.1061,
 0.0598,  0.0353, -0.1610,
 0.0165,  0.0835,  0.0704,
-0.0079, -0.0982,  0.0187,
 0.2331, -0.1929,  0.0684,
-0.0507,  0.1476, -0.0886,
-0.0275,  0.1658,  0.0697,
-0.1123, -0.0069, -0.0851,
-0.0377, -0.0917, -0.0629,
-0.0420,  0.0506,  0.1111,
 0.1086,  0.1351, -0.0851,
 0.0466,  0.2750,  0.0185,
-0.0208,  0.2090,  0.0271,
 0.0217, -0.0548,  0.0078,
-0.0609,  0.1029, -0.1641,
 0.1392,  0.0115,  0.0317,
-0.0570,  0.1060,  0.1814,
-0.2015, -0.1301,  0.1082,
 0.2452, -0.1815, -0.0046,
 0.0103, -0.0466, -0.0895,
 0.0158, -0.0594, -0.1386,
-0.0073, -0.0719, -0.0716,
 0.1308, -0.0206,  0.0511,
-0.0437, -0.0763,  0.0287,
 0.0493, -0.1239,  0.0219,
-0.0041,  0.0373,  0.0262,
 0.0078, -0.0249, -0.0284,
 0.0598, -0.0205, -0.0276,
 0.0115, -0.1778, -0.0395,
 0.1673, -0.0036,  0.2334,
 0.0706, -0.0694,  0.0177,
 0.1123, -0.0043,  0.0716,
-0.0894, -0.1609,  0.0334,
-0.0046, -0.2006, -0.0977,
-0.0127,  0.1198, -0.0339,
-0.0283,  0.1354,  0.1637,
-0.1696,  0.0187, -0.2621,
 0.0496,  0.2834,  0.0423,
 0.1126,  0.3962,  0.1660,
-0.0750,  0.1955,  0.0590,
-0.1088, -0.1146, -0.1219,
 0.1360,  0.1524,  0.0498,
-0.1151,  0.0219, -0.0063,
-0.0821,  0.0247, -0.1065,
 0.1153,  0.2085,  0.0618,
-0.0383,  0.0527, -0.2067
}
,)"
R"(
{
 1.8014e-01,  2.1908e-01, -2.1088e-03,
 1.7345e-01,  2.7654e-01,  1.3607e-02,
 1.1363e-01,  9.9105e-02, -6.5730e-02,
-3.5679e-02,  9.6072e-03,  4.0721e-02,
-1.8771e-02, -2.3484e-04, -1.0230e-02,
 1.6965e-02, -1.3032e-02, -6.3906e-02,
-4.5686e-02, -3.6733e-02, -4.8873e-02,
 4.0752e-02,  2.1615e-02, -1.4822e-02,
 1.1689e-01,  3.0153e-02, -5.0163e-04,
-7.0394e-03, -1.2387e-01, -8.9243e-02,
-1.8312e-01, -1.3868e-01, -6.2618e-02,
-8.1627e-02, -2.0480e-01, -3.0740e-01,
 4.4296e-02,  3.8572e-02,  4.3754e-02,
 1.7538e-01,  5.3284e-02, -7.5663e-03,
 1.9670e-01, -1.2397e-01, -1.6266e-01,
 1.4575e-01, -5.7771e-02,  2.7619e-02,
 2.2757e-02, -4.8910e-01, -2.6201e-01,
 3.6513e-02, -2.0704e-01, -1.3225e-01,
-6.7533e-02,  1.1289e-02,  7.1316e-02,
-7.6847e-02,  6.8128e-02,  7.4717e-02,
 1.1269e-01,  2.9978e-02,  3.2132e-02,
-5.4557e-02, -4.4599e-02,  4.1835e-02,
 5.7964e-02, -2.1246e-03,  1.5007e-01,
 1.8432e-01,  1.1463e-01,  2.2691e-01,
 9.6166e-02,  4.7887e-02, -3.8399e-02,
 5.8153e-02, -2.0255e-02, -1.1362e-01,
 2.6402e-02,  2.5562e-02,  1.9096e-02,
 1.1588e-01,  1.4540e-01,  1.1948e-01,
 1.0360e-01,  5.9083e-02,  1.9263e-01,
 1.6953e-01,  2.7390e-02,  9.7883e-02,
 1.5059e-01,  6.7593e-02, -4.5843e-03,
 8.7031e-02, -2.0926e-03, -6.3056e-02,
-6.6960e-02, -5.2056e-02, -7.3570e-02,
 1.4361e-02,  1.1059e-01, -4.9720e-02,
 4.4270e-02,  3.9995e-02,  4.3101e-03,
-1.1042e-01,  4.5028e-02, -8.9124e-02,
-1.2906e-01, -7.6972e-02, -6.5449e-03,
-1.9269e-01,  2.8349e-01,  1.1573e-01,
-1.7983e-01,  9.7615e-02,  9.4003e-03,
-4.7802e-02, -1.5889e-01, -1.2693e-01,
 7.4717e-02,  2.8655e-01, -7.2637e-02,
 1.5837e-02,  8.7125e-02, -1.2198e-01,
-1.7754e-02, -5.6443e-02, -9.8661e-03,
 6.3040e-02,  2.0249e-02, -3.5368e-02,
 9.7756e-03,  2.6760e-02, -5.5172e-02,
-1.0406e-02,  4.8313e-02,  2.4717e-02,
-5.2851e-02,  6.8496e-02, -2.5933e-02,
 4.5932e-02,  5.9892e-02,  1.9200e-02,
-5.1316e-40, -5.1811e-40, -1.5144e-40,
-6.7758e-38, -5.4608e-40, -3.9680e-40,
-1.9155e-39,  2.0423e-41,  1.5256e-41,
-2.5559e-08, -3.2461e-08, -2.6821e-08,
-3.6885e-08, -4.6896e-08, -3.9086e-08,
-3.4305e-08, -4.4160e-08, -3.7187e-08,
-3.7416e-40,  3.6550e-40,  5.0727e-40,
-1.6722e-40,  3.9228e-40,  5.4548e-40,
-5.7512e-40, -2.8156e-40,  9.4571e-41,
-4.7040e-40, -1.6974e-40,  6.3849e-40,
-3.7322e-40,  2.6014e-40,  2.3080e-40,
-2.8395e-40, -3.7116e-40,  4.4393e-40,
 1.1597e-40,  4.3291e-40,  3.8219e-40,
 3.3393e-40,  3.1747e-40, -1.8400e-36,
-5.5215e-40,  1.7648e-40, -1.6540e-35,
-3.0953e-40,  5.3063e-40, -1.6454e-40,
 2.1341e-40,  2.0790e-40, -3.0226e-40,
-2.6807e-40, -1.6601e-40,  5.1829e-40,
-1.8897e-40, -4.5956e-41,  5.3784e-40,
-2.5661e-40, -2.1726e-40,  1.2010e-40,
 1.8263e-41,  1.1214e-40, -3.7693e-40,
-4.2596e-40,  1.8854e-40,  5.5010e-40,
-6.6262e-40, -4.8808e-40,  3.3123e-40,
 5.9379e-41,  2.3249e-40,  4.4504e-40,
-8.4836e-04, -8.4397e-04, -5.8640e-04,
-8.3506e-04, -8.0192e-04, -5.3901e-04,
-8.3539e-04, -7.8069e-04, -4.8720e-04,
-3.4706e-04, -4.4640e-04, -5.2353e-04,
-4.4518e-04, -5.3374e-04, -5.2734e-04,
-5.8780e-04, -5.8730e-04, -5.4362e-04,
-5.2452e-04, -5.4578e-04, -5.6266e-04,
-4.2387e-04, -4.4643e-04, -4.8936e-04,
-3.5880e-04, -3.7886e-04, -4.1998e-04,
-2.4479e-04, -4.0736e-04, -3.1189e-04,
-3.4922e-04, -4.0173e-04, -2.5042e-04,
-5.7091e-04, -5.2665e-04, -2.3293e-04,
-2.8505e-04,  9.7283e-05,  3.1209e-04,
-2.7463e-04,  1.8704e-04,  4.4351e-04,
-9.1436e-05,  3.2602e-04,  5.7573e-04,
-4.0112e-04, -4.2566e-04, -2.4300e-04,
-9.9362e-05, -6.5499e-05,  3.2872e-05,
 1.1584e-04,  2.3417e-04,  3.4427e-04,
-7.5767e-05,  3.9768e-06,  6.2201e-05,
 2.3151e-05,  2.5595e-04,  3.4038e-04,
-1.3871e-05,  3.0295e-04,  4.4170e-04,
-1.7802e-04, -4.5376e-04, -5.1847e-04,
-5.0687e-04, -5.5837e-04, -2.5917e-04,
-5.3992e-04, -7.1375e-04, -4.8728e-04,
-1.7543e-01, -3.4151e-01, -3.2619e-02,
-1.9701e-02, -1.5494e-01, -1.6534e-01,
 3.5632e-02, -1.0897e-01, -3.8379e-02,
-6.1420e-02, -1.0735e-01,  1.4730e-01,
 7.4386e-02, -1.0487e-01,  7.9646e-02,
 1.7130e-02,  4.4391e-02, -5.1959e-03,
 4.5682e-02, -1.1543e-01,  9.4035e-03,
-3.4376e-01, -1.1961e-01,  1.0099e-01,
 1.1335e-01,  7.5840e-02,  1.0675e-01,
 4.9539e-02,  8.7406e-02,  4.4951e-02,
 1.8111e-01,  2.6406e-01, -1.5924e-02,
-1.1464e-01,  8.4579e-04, -6.6811e-02,
-8.9635e-03,  1.8236e-03,  3.6561e-02,
-7.0281e-02,  2.9717e-01,  3.1836e-02,
-1.3647e-01, -6.5627e-02,  9.3063e-02,
-2.1851e-01, -6.0226e-02, -1.0326e-01,
 5.3441e-02,  1.9103e-01, -5.7999e-02,
-3.3512e-02,  1.5496e-01, -1.1111e-01,
 2.3256e-03, -1.5004e-01, -9.1248e-02,
-9.7706e-02,  1.9549e-01, -1.5403e-01,
-1.5327e-01,  8.3335e-02,  5.6111e-03,
-1.5707e-01,  8.0277e-03, -7.3955e-02,
-1.4111e-01, -1.3548e-01, -1.0563e-01,
 2.3054e-01, -2.1822e-02, -6.6938e-03,
-1.0259e-01,  4.3577e-02, -1.7630e-01,
 1.6484e-01,  4.2413e-01,  6.9475e-02,
-2.4705e-01,  2.5757e-01, -9.5611e-02,
 1.0236e-01, -3.4820e-02, -6.8818e-03,
-1.1434e-01, -3.1800e-01,  2.1337e-02,
-1.9939e-01, -2.6532e-01,  7.3361e-02,
 6.5939e-02,  9.5812e-02, -7.0156e-02,
-1.6249e-02, -1.5927e-02, -1.1189e-01,
-9.3936e-03, -1.0933e-01, -2.9399e-02,
-2.8752e-02, -4.5613e-02, -1.2718e-02,
 3.8781e-01,  2.6776e-01, -1.0373e-02,
-2.3927e-02, -6.4398e-02,  9.9117e-02,
-6.0732e-02, -5.5917e-03,  5.1716e-02,
-1.4168e-01,  1.7661e-01, -5.5893e-02,
-3.0419e-01, -3.5537e-01,  2.1978e-01,
-1.8610e-01, -5.7743e-03,  3.2649e-02,
 1.9975e-01,  1.6508e-01,  1.3808e-02,
 1.0733e-01,  1.4722e-01,  5.8671e-02,
 6.4940e-02,  1.6114e-01,  3.9697e-02,
 1.1530e-01,  2.4021e-01, -2.1669e-01,
 6.0220e-02,  2.0257e-01, -1.5227e-01,
-6.1096e-02,  6.6511e-02, -1.3858e-01,
-6.5275e-02,  1.0891e-01,  8.2048e-02,
-6.7907e-02,  2.2863e-02, -1.0322e-01,
 1.6542e-01, -1.4436e-01,  6.4125e-02,
-1.0378e-01, -3.2346e-01, -1.5123e-02,
 3.8758e-03,  1.1006e-01, -4.4325e-02,
-1.0102e-01, -3.7699e-02,  9.2472e-02,
-6.8972e-02, -1.2308e-02,  1.6478e-01,
 3.4351e-02, -1.7461e-02,  1.0301e-01,
-2.7125e-01, -5.6730e-02, -2.5989e-01,
-3.0163e-01, -1.4826e-01, -3.4955e-01,
-1.6259e-01, -1.6708e-01, -2.7964e-01,
-6.7134e-02, -2.2385e-01,  2.1776e-01,
-1.1351e-02, -3.7861e-01,  1.8687e-01,
 4.0551e-02,  8.1943e-02,  1.0866e-01,
 1.0273e-01,  1.1844e-01, -1.1852e-01,
 2.6758e-02, -8.5806e-02,  5.9444e-02,
-5.1627e-02,  7.1636e-02,  2.2841e-01,
-3.7242e-03,  2.9723e-01,  1.1918e-01,
 8.4994e-02, -3.5747e-01,  3.6148e-02,
 9.9705e-02, -1.3736e-01, -6.0080e-02,
 1.2370e-01,  5.0668e-02, -6.0246e-02,
 6.0562e-02, -3.5068e-01, -3.2645e-01,
 9.1020e-04,  6.6203e-02, -1.0770e-01,
 1.9434e-02,  3.0018e-01,  2.8018e-01,
 1.4021e-01,  2.7481e-01,  2.2868e-01,
 4.8540e-02,  1.7719e-01, -4.5834e-02,
-9.6349e-02, -2.3008e-02, -1.4497e-01,
 4.3053e-02, -1.0161e-01,  2.8750e-02,
-1.2594e-01, -1.0388e-02, -4.3966e-02,
 7.5993e-02, -7.1609e-02,  1.4624e-02,
 4.1110e-02,  7.1258e-02, -2.9109e-02,
-5.8698e-03,  1.2389e-01,  4.7648e-02,
-6.1585e-04, -4.4556e-02, -2.3373e-02,
-4.4883e-02, -7.7722e-02, -7.3635e-02,
-2.7750e-02, -1.5117e-03, -8.7368e-02,
 2.5113e-02,  7.7490e-02,  2.9024e-02,
 1.5426e-01,  2.5472e-01,  4.8057e-02,
-1.1969e-01, -1.1487e-01, -1.1802e-01,
-4.7392e-02, -4.2226e-02,  3.1968e-02,
-2.6717e-01, -5.0206e-02,  8.1946e-04,
-4.0426e-02,  1.4373e-01, -3.3121e-03,
-4.5292e-02, -2.4538e-02,  1.0377e-01,
-1.7780e-02,  2.0058e-01, -2.4343e-02,
-1.1714e-02,  1.5984e-01, -1.2638e-01,
 6.4655e-02,  3.7703e-02,  3.7970e-02,
 9.1864e-03,  1.1468e-01, -6.2760e-04,
-1.4812e-01,  6.5670e-03,  1.0765e-01,
 1.5023e-01, -7.0594e-02, -1.3924e-01,
 3.6016e-02, -3.9078e-02, -3.8950e-02,
 1.8735e-02, -1.5573e-01, -1.2456e-01
}
,)"
R"(
{
 4.8634e-02, -1.3617e-01,  6.1231e-02,
-7.0235e-02, -6.4110e-01,  1.5985e-01,
 8.6151e-02,  1.1847e-01,  1.3819e-01,
-3.6017e-04, -3.2273e-02, -8.5485e-02,
-7.0804e-03,  2.1751e-01,  7.2575e-03,
-8.3606e-02, -1.4885e-01, -1.2702e-01,
 4.0848e-41,  8.0934e-40, -1.8889e-40,
-3.9103e-40, -7.4709e-40,  3.8377e-40,
-2.4159e-40, -4.7610e-40,  7.7359e-40,
-8.6217e-05, -5.9763e-05, -4.0558e-05,
-7.4966e-05, -4.7074e-05, -3.1656e-05,
-9.8390e-05, -6.6833e-05, -4.7669e-05,
 3.5375e-02,  2.8660e-02,  4.1277e-02,
 1.6289e-01, -3.2199e-01, -1.7845e-02,
 2.4659e-01, -3.9618e-02,  4.1065e-03,
 2.7267e-02,  8.6819e-02,  9.5070e-02,
-7.2700e-02, -2.8826e-01,  1.1750e-03,
 2.5259e-02,  2.4681e-03,  6.4737e-02,
 7.3023e-03,  2.9631e-02,  1.0820e-02,
-2.1400e-02,  5.4244e-01,  1.5639e-01,
-1.7561e-01,  4.8947e-01, -8.8305e-02,
 6.5073e-02,  3.4922e-01,  1.3483e-01,
 1.4506e-01, -2.5472e-01, -7.2894e-02,
 4.5945e-02,  1.4040e-01,  1.2148e-01,
-2.6932e-01, -1.1518e-01, -9.3158e-03,
-2.3961e-01, -1.2479e-01, -8.9796e-02,
 1.8688e-02, -4.9267e-02,  7.7189e-02,
-7.3691e-02,  7.8186e-03,  1.3761e-02,
-1.5689e-01,  3.1138e-02,  3.9231e-02,
-4.3607e-03,  2.0813e-01,  5.5635e-02,
-6.7000e-41,  9.8995e-41,  3.0043e-40,
 6.7190e-40,  4.0827e-40,  7.6057e-40,
 4.2208e-40,  8.1141e-40, -3.3569e-40,
 1.0179e-03,  5.1543e-04,  3.8076e-04,
 7.3507e-04,  4.5432e-04,  3.7410e-04,
 9.3014e-04,  6.7365e-04,  6.0051e-04,
-5.1998e-02,  6.5768e-02,  3.1603e-02,
-3.0198e-02, -3.1692e-02, -6.9299e-02,
 1.7672e-02,  2.3766e-01,  5.7877e-02,
-5.7944e-02,  1.2624e-01, -1.4396e-01,
-4.1542e-02,  6.5110e-01,  1.0942e-01,
-1.3133e-01,  5.0538e-02, -2.7371e-02,
-3.7515e-02,  2.8703e-02,  1.2382e-03,
 3.8542e-01, -2.2754e-02,  3.4459e-02,
 3.0545e-01, -5.3817e-01, -2.1389e-03,
 1.3888e-02, -2.2775e-01, -6.3692e-02,
-1.8430e-01,  5.8452e-02,  4.5764e-02,
-8.5045e-02, -1.7060e-01, -1.8565e-02,
-2.0384e-02, -3.3018e-02, -5.1135e-02,
-4.5789e-02, -1.8105e-01,  3.5419e-02,
-5.0081e-02,  8.7719e-02,  1.0373e-01,
-1.0033e-02,  7.0530e-02, -7.8012e-03,
 8.4042e-02,  1.1982e-01, -9.6046e-02,
-6.4009e-02, -1.0711e-01, -1.3523e-01,
 1.8868e-41, -7.0039e-40, -7.2568e-40,
 1.7408e-40, -7.8143e-40, -6.8130e-40,
-6.3142e-40, -6.2560e-40, -7.4238e-40,
 2.6297e-04,  7.0014e-05, -4.0981e-04,
 2.6263e-04,  4.2811e-05, -4.9950e-04,
 3.9795e-04,  1.2615e-04, -4.7660e-04,
 7.5933e-02,  2.6295e-02,  2.7984e-02,
-5.5914e-03, -8.7981e-02, -9.2618e-02,
 4.2725e-02, -3.1210e-01,  1.3412e-01,
 5.2683e-02,  3.9891e-01,  2.9150e-02,
-6.6090e-02,  2.9455e-01, -1.9710e-01,
 1.4546e-02, -2.5572e-02,  8.1125e-02,
 1.2271e-01,  1.6097e-01,  4.5644e-02,
 3.6101e-02, -1.7174e-02,  6.6110e-02,
 1.5078e-01,  4.5180e-01,  7.7154e-02,
-5.9725e-02,  1.0185e-01,  1.1363e-03,
 6.7791e-02,  1.7696e-02,  5.2638e-02,
 3.3051e-02, -8.4049e-02,  1.4380e-01,
 1.8744e-02, -2.0940e-01, -2.1424e-01,
-2.1329e-01, -1.3154e-01, -3.2572e-01,
 1.1292e-01,  1.2361e-02, -1.5506e-01,
-1.0362e-02,  1.9955e-02,  4.2639e-02,
-2.1952e-02, -2.4682e-02, -2.4453e-02,
-2.5606e-02, -3.3580e-02, -3.6340e-02,
-5.0830e-40,  6.3797e-40, -5.2775e-40,
-7.7988e-40, -7.4579e-40, -5.1901e-40,
-3.8275e-41, -5.7607e-40, -1.3656e-40,
 2.7164e-04,  5.9977e-04,  8.6886e-04,
 3.0116e-04,  7.0106e-04,  1.0248e-03,
 2.9177e-04,  6.4748e-04,  9.4825e-04,
 6.6310e-02,  1.5240e-02, -5.3044e-02,
 1.2545e-01,  5.0582e-02,  2.7358e-02,
 1.9338e-01,  1.1377e-01,  4.6110e-02,
-3.1997e-02,  1.5171e-02, -4.9372e-02,
 5.4615e-04,  1.7262e-01, -2.2081e-01,
 8.4871e-02,  1.7824e-02, -3.6429e-02,
 4.2821e-02, -1.0055e-01,  4.8927e-02,
 1.2524e-01,  5.8859e-02, -2.0980e-02,
 2.2897e-01,  1.7594e-01,  3.4239e-02,
 1.0915e-01,  1.2088e-01,  1.0151e-01,
 6.8449e-03, -1.5546e-01,  1.2024e-01,
 4.9036e-02, -1.2245e-01,  4.6713e-02,
 7.5083e-03, -4.8084e-02,  9.7731e-03,
 4.8779e-02,  3.1848e-02, -9.3517e-02,
 6.4595e-02,  3.9337e-02, -7.2343e-02,
 3.9519e-02,  4.1867e-02, -5.0485e-02,
 2.5257e-02,  1.4071e-01,  1.3606e-01,
 1.7481e-01,  2.0210e-01,  1.7241e-01,
-7.6295e-40, -7.8460e-40, -4.1806e-41,
-7.9994e-40, -7.3271e-40, -6.2665e-40,
-7.9602e-40, -7.0226e-40, -7.4131e-40,
-4.5544e-04, -5.2379e-04, -7.0755e-04,
-3.3807e-04, -3.8123e-04, -5.3222e-04,
-3.1771e-04, -3.4586e-04, -4.8784e-04,
-3.5257e-02, -1.1866e-02,  1.9717e-02,
-6.0777e-02, -7.3127e-03, -3.2825e-02,
-1.4952e-01,  3.2117e-01, -6.3786e-02,
-1.0255e-02,  1.2961e-01, -8.6823e-02,
 1.6994e-01,  4.7491e-01,  2.7135e-01,
 2.8538e-03,  1.5572e-01, -3.3736e-02,
 8.5996e-02, -1.0176e-02,  2.6629e-02,
 7.3362e-02, -7.7525e-03,  5.6261e-02,
 1.0819e-01, -2.5863e-01, -5.7146e-03,
-7.1781e-02,  2.8376e-03,  7.8298e-02,
 1.3183e-01,  2.7149e-02, -9.9786e-02,
 9.0491e-02,  8.7938e-02, -2.1882e-02,
 4.1396e-03, -4.5816e-02, -7.8892e-02,
-6.3855e-03,  1.7502e-01,  1.2053e-01,
 1.2492e-01,  6.1258e-02, -4.0516e-02,
-4.5409e-02, -4.5877e-02, -7.6414e-02,
-1.0573e-02, -1.2517e-01, -4.3991e-02,
-2.6447e-02, -9.5478e-02, -2.4735e-02,
-4.6548e-41, -1.6443e-40, -3.1221e-40,
-3.2675e-40, -2.7265e-40, -3.1190e-40,
-2.2065e-40, -2.5407e-40, -6.9511e-40,
-1.2727e-04, -2.6585e-04, -3.5516e-04,
 3.4272e-05, -1.6810e-04, -3.1677e-04,
-5.5355e-05, -2.9924e-04, -4.3692e-04,
-5.6428e-02,  1.0771e-01,  1.0185e-01,
 2.2948e-01, -7.8744e-02,  6.0768e-04,
-2.2355e-03, -2.0128e-03, -5.7317e-03,
-7.1232e-03,  1.0297e-01,  1.6872e-01,
 1.9194e-01, -1.1578e-01,  1.0732e-01,
-8.6952e-02,  3.2901e-02, -6.6658e-03,
 7.3979e-02,  8.3875e-02, -7.6372e-03,
 1.9577e-01,  2.7391e-01,  4.5275e-02,
 1.5610e-01,  2.3802e-01,  1.6555e-02,
 1.3814e-01,  1.2870e-01,  9.1626e-02,
-4.6890e-02, -8.8734e-02,  7.8866e-02,
 1.0027e-01,  2.2139e-01,  1.0050e-01,
-6.5845e-02, -1.0990e-01, -6.9896e-02,
 4.1687e-02,  3.0631e-02, -8.8441e-02,
-1.1868e-01,  1.0836e-02,  2.5873e-02,
-1.7114e-02,  7.6295e-02,  1.5439e-02,
-2.4271e-02,  5.8538e-02,  9.8190e-02,
 4.9742e-02,  8.7807e-02,  6.5871e-02,
-7.2669e-40, -7.5936e-41, -7.4975e-40,
-1.6984e-42, -1.7334e-40, -8.4954e-41,
-2.1556e-41, -1.5374e-40, -1.5515e-40,
-6.2626e-04, -7.2727e-04, -8.1665e-04,
-5.6584e-04, -6.1190e-04, -6.9584e-04,
-5.6278e-04, -5.8554e-04, -6.3554e-04,
 8.1550e-02, -4.1817e-03,  1.2301e-02,
-4.5800e-02,  4.6708e-02, -8.7972e-02,
-2.9880e-01,  2.6456e-01,  3.9363e-03,
-3.0939e-02, -1.9921e-01, -3.8689e-03,
-8.6803e-02,  3.4857e-01, -1.0201e-01,
 2.1597e-02,  1.4380e-02,  4.3448e-02,
 7.1195e-02,  1.4980e-01,  3.8079e-02,
-1.2678e-01, -8.1274e-02, -4.3445e-02,
 5.2482e-02, -1.8763e-01,  1.1557e-01,
-9.4614e-02,  5.4415e-02, -3.1485e-02,
-3.6451e-02,  1.4379e-01,  5.2291e-02,
-9.2069e-02,  9.5675e-02, -5.8433e-02,
 7.5768e-03, -7.1280e-02, -1.4576e-01,
-1.4671e-01, -1.2446e-01, -1.5207e-01,
-5.4368e-02,  3.8303e-02, -8.1794e-02,
 2.0492e-02,  4.0910e-02,  1.1379e-02,
 3.1582e-02,  3.6039e-02, -4.4040e-03,
 1.7540e-02,  1.4097e-04, -6.4367e-02,
-7.9553e-40, -5.3941e-40, -7.1912e-40,
-5.8099e-40, -6.8315e-40, -6.6012e-40,
-7.6242e-40, -5.4784e-40, -7.0267e-40,
-2.9197e-04, -2.1994e-04, -1.9501e-04,
-2.6516e-05, -1.2642e-05, -8.4345e-05,
 1.6763e-04,  1.1268e-04, -5.4516e-05,
-3.8007e-03, -6.8765e-02, -9.5716e-02,
 6.3091e-02, -8.1971e-02, -9.2895e-02,
-6.8353e-03,  7.3639e-02,  1.3505e-01,
 9.0083e-02,  2.4352e-01,  3.9708e-02,
-5.4051e-02, -6.8748e-02, -1.8937e-01,
-1.9808e-03, -7.1337e-02, -2.8316e-02,
 8.1504e-02,  8.3226e-03,  6.9013e-03,
 9.4393e-02,  5.9322e-02,  5.5023e-02,
 1.0236e-01, -4.0205e-02,  3.5172e-02,
 6.5381e-02,  4.9075e-02, -5.3931e-02,
 4.3961e-02,  9.0223e-03, -4.1678e-02,
-6.4262e-02, -5.0304e-02, -9.3597e-02
}
,)"
R"(
{
 3.8496e-01,  1.4287e-01,  3.4530e-02,
-5.5398e-01, -6.0381e-02,  1.2078e-02,
 7.9983e-02,  2.1478e-01, -5.7915e-02,
-1.4020e-01, -2.6914e-02,  1.5915e-02,
 1.2371e-01,  2.5496e-01, -2.9867e-02,
 1.3269e-02, -9.9596e-02, -2.3173e-01,
 5.1471e-02, -4.5507e-01, -7.7620e-02,
-5.1328e-02, -1.9808e-02, -4.7051e-02,
 3.0573e-02,  7.8762e-02, -7.2627e-02,
 6.8690e-02, -4.0125e-02,  5.6657e-02,
 8.0208e-02, -2.0075e-02,  1.4019e-01,
-5.7959e-02, -7.3152e-02,  2.0202e-02,
-8.8702e-02, -1.9911e-01, -1.5570e-01,
 2.8401e-02,  5.8802e-02,  1.3050e-01,
 2.1905e-02, -3.4298e-02,  4.0447e-02,
 1.0184e-01, -9.0101e-02, -9.2770e-02,
 1.1713e-02, -3.2514e-01,  1.9393e-01,
-9.4227e-02,  2.7053e-01, -9.7233e-02,
-1.0478e-01,  6.0652e-02,  8.3399e-02,
 1.1104e-01,  2.9008e-01,  4.9208e-02,
-1.5414e-02,  3.1718e-02, -7.9083e-02,
-5.2358e-03,  9.0101e-02,  5.2973e-02,
 5.5527e-02, -1.6599e-02, -8.5167e-02,
-5.1018e-02,  7.2243e-03, -9.5684e-02,
-5.0608e-02, -6.7864e-02, -8.9496e-02,
-2.4348e-01,  2.7477e-01, -1.7588e-01,
 1.3927e-01,  5.5502e-02, -1.3370e-02,
-4.3509e-02, -2.1511e-01, -5.9070e-02,
 1.0293e-01,  4.2678e-01, -8.7527e-02,
-6.8546e-02, -5.6296e-02, -8.7962e-02,
-8.6130e-02,  9.2069e-02,  7.2303e-02,
 2.4365e-02,  2.1988e-01, -7.9408e-03,
-3.0063e-02,  1.1554e-01, -5.0311e-02,
 1.0605e-02,  5.4598e-02,  1.3826e-02,
-1.4342e-02,  1.5353e-01, -5.3974e-03,
 1.5583e-01, -6.0889e-02, -1.5772e-02,
-2.5956e-02, -3.5285e-01, -2.0338e-01,
 2.6011e-01,  2.2737e-01, -1.4693e-01,
-7.7964e-02,  1.0053e-01, -5.4278e-02,
-3.0668e-02,  3.4556e-02, -3.4321e-02,
 7.8695e-02, -2.2357e-01,  9.5733e-02,
 1.7483e-01, -1.5153e-01, -1.8262e-03,
 4.7605e-02, -2.2834e-01,  4.6383e-02,
 1.5701e-01,  3.2264e-01,  1.0334e-02,
 6.3351e-02,  1.1340e-01,  8.3478e-02,
 6.4196e-02,  3.3460e-02,  8.8473e-02,
 5.4663e-02, -1.7665e-03, -4.1935e-02,
-6.1346e-03, -5.4463e-02, -6.2960e-02,
 2.8159e-02,  2.9903e-02,  9.2429e-03,
-3.0041e-02, -9.7783e-02, -4.9500e-02,
 9.5350e-02, -7.9143e-02, -1.3244e-01,
-6.5129e-02,  1.4568e-01,  6.6843e-02,
 1.5241e-01, -7.8736e-02,  1.0721e-01,
-5.9015e-02,  1.5320e-01,  3.0796e-01,
-5.4266e-03, -6.0804e-02,  3.7326e-02,
 7.4844e-02,  4.8340e-02,  1.5251e-01,
 3.8158e-02,  1.2087e-01, -8.9003e-02,
-5.8369e-02, -7.3813e-02,  1.2240e-02,
-4.5106e-03,  7.4580e-02,  1.2042e-01,
 4.1959e-02,  1.4529e-01,  5.3636e-03,
-4.9708e-03, -1.0775e-02, -5.9374e-02,
 1.5358e-02,  1.7277e-02, -1.5412e-01,
 8.1647e-02,  3.3503e-02, -8.1934e-02,
-1.5807e-02, -1.0001e-02, -1.0059e-02,
-9.0493e-03, -7.8954e-02,  4.3891e-02,
-9.3815e-03,  3.2241e-02,  4.7962e-02,
-7.2252e-03,  7.9324e-02,  2.0662e-02,
-5.7710e-02, -5.1142e-02, -1.4296e-01,
 2.1501e-02, -1.9518e-02, -2.7658e-02,
 1.4983e-01,  8.5447e-02,  7.2092e-04,
 1.1275e-01,  6.1131e-02,  5.7955e-02,
 1.5624e-02,  2.7225e-01,  1.1716e-01,
-1.6322e-04, -1.3368e-04, -1.5575e-04,
-1.0525e-04, -1.0765e-04, -1.5306e-04,
-8.9692e-05, -1.0857e-04, -1.7316e-04,
-1.8015e-03, -1.3733e-03, -3.9154e-04,
-1.8453e-03, -1.4238e-03, -4.4163e-04,
-1.5511e-03, -1.1131e-03, -2.0087e-04,
-2.4082e-03, -2.2576e-03, -1.9231e-03,
-2.4913e-03, -2.4136e-03, -2.1678e-03,
-2.5057e-03, -2.4650e-03, -2.2732e-03,
-2.3901e-05, -1.5870e-05, -5.8255e-06,
-1.5163e-05, -1.2370e-05, -6.0712e-06,
-1.3098e-05, -1.1132e-05, -5.7866e-06,
-5.9760e-03, -5.9998e-03, -6.0295e-03,
-5.9962e-03, -6.0100e-03, -6.0277e-03,
-6.0003e-03, -6.0059e-03, -6.0148e-03,
-3.2764e-05, -2.9574e-05, -2.8001e-05,
-1.0846e-05, -1.1569e-05, -1.4282e-05,
-1.6255e-06, -2.5666e-06, -4.7808e-06,
-5.1999e-03, -5.2334e-03, -5.2847e-03,
-5.2057e-03, -5.2283e-03, -5.2713e-03,
-5.2195e-03, -5.2321e-03, -5.2633e-03,
-3.0782e-06, -9.2118e-06, -1.6177e-05,
-1.6382e-06, -6.9559e-06, -1.4245e-05,
-1.1471e-06, -6.5984e-06, -1.4903e-05,
 7.7574e-02, -1.2866e-02,  4.1348e-03,
-6.7298e-02, -1.3691e-01,  6.4079e-02,
 3.7962e-02,  8.7737e-02, -4.1046e-02,
-2.8471e-02,  1.7647e-01,  6.4232e-02,
 1.2316e-01,  3.6800e-01, -1.5740e-01,
-6.0839e-02,  1.5449e-02, -1.0761e-01,
-6.6869e-02, -1.2867e-01, -4.0195e-02,
-4.9651e-02, -5.5500e-02, -2.5879e-02,
 2.0179e-02,  6.8467e-02,  2.6575e-02,
-6.7728e-04, -7.6269e-02,  2.3470e-02,
 7.1869e-02, -1.1855e-01, -2.1067e-02,
 1.3263e-01, -3.2957e-02, -3.4365e-03,
 8.1936e-02,  1.3073e-01,  1.1477e-01,
 1.2429e-01,  1.6129e-01,  1.6251e-01,
 1.5476e-02,  3.2862e-02,  2.1999e-02,
-2.9189e-02, -3.3615e-02,  5.5616e-04,
-2.4059e-02, -9.6181e-03, -4.1175e-02,
-6.3680e-04, -9.6559e-02, -9.1448e-02,
 3.0238e-02,  1.2534e-01,  1.5256e-02,
-4.2118e-02,  1.5723e-01,  2.6929e-03,
 1.9873e-02,  5.3050e-02, -1.0153e-03,
 2.0634e-02,  9.2825e-03, -6.8027e-03,
 3.1335e-03, -7.7443e-03, -1.8307e-02,
 7.9974e-03, -1.0283e-03, -6.2520e-03,
 4.5050e-02,  9.9504e-02, -1.3404e-01,
-6.7271e-01, -5.7290e-02,  2.6919e-02,
 2.3673e-01,  2.4688e-02, -2.0227e-02,
 5.1389e-02, -3.9810e-02, -8.9700e-02,
 2.8445e-02,  3.9136e-01, -1.1508e-01,
-1.0449e-01, -6.2005e-02,  6.5721e-02,
-1.9123e-01, -4.2613e-02,  3.5371e-02,
 1.9207e-01,  8.7916e-02,  4.8089e-02,
-5.7912e-02,  1.0014e-01, -9.4659e-02,
 1.1240e-02, -6.2254e-03,  1.3399e-01,
 1.6483e-01, -3.5079e-01,  1.1612e-02,
 2.9215e-01,  5.6875e-02,  6.9505e-02,
 1.3721e-02,  1.2607e-01,  2.6426e-02,
-2.0529e-01,  2.1768e-01,  2.1232e-01,
-6.3574e-02,  2.3504e-02, -1.0811e-01,
-1.3470e-02, -3.6446e-02, -5.4379e-02,
-1.3257e-01, -8.3412e-02,  3.7745e-02,
 5.8778e-02, -2.6060e-01,  3.8262e-02,
-4.3689e-03, -6.6703e-02, -2.2025e-01,
-9.0961e-02,  1.3855e-01,  3.4573e-04,
-2.9613e-01, -3.6138e-02, -1.3827e-01,
 4.5896e-02, -5.3871e-02, -1.0037e-01,
 1.8457e-01,  1.0338e-01, -5.7306e-02,
 5.5510e-02, -9.4938e-02, -5.6527e-05,
 1.6372e-01, -3.3854e-02,  5.6332e-02,
-4.0251e-01, -5.9428e-02, -9.1470e-02,
-1.5921e-02, -5.7948e-02,  8.1682e-03,
-3.7833e-03,  1.6293e-01,  5.3784e-02,
 1.1053e-01, -1.3867e-01,  2.6772e-02,
-1.3133e-02,  3.7614e-01,  3.6361e-03,
-1.4205e-01,  3.1312e-02, -9.9928e-02,
-1.5755e-01,  4.2016e-01,  9.4065e-02,
 2.7536e-02,  1.2620e-01, -1.4894e-01,
-4.2137e-02, -9.8700e-02, -1.7479e-01,
 4.5836e-02,  5.3893e-02, -1.0138e-01,
 8.3609e-02,  2.1849e-02, -1.0648e-01,
 7.4801e-02, -1.2671e-01, -1.5007e-02,
 2.7440e-01, -3.1351e-01,  6.5787e-02,
-6.7820e-02,  1.6312e-01, -1.3254e-02,
-2.5770e-02, -2.0041e-02,  5.8243e-02,
 1.6055e-02,  1.1971e-02, -4.6112e-02,
-1.6276e-01, -1.5313e-02, -7.9826e-03,
 9.1668e-02,  9.7722e-02,  1.3754e-01,
-7.4817e-02, -4.1923e-01, -1.2337e-01,
 1.3472e-01, -4.0745e-02, -5.4055e-02,
-1.2943e-02,  4.8796e-02,  4.2007e-02,
 9.4668e-02,  8.6149e-02,  1.2362e-01,
 7.0637e-02,  2.3565e-01,  1.4582e-01,
 5.6904e-02, -8.2166e-02,  1.0563e-01,
 9.3969e-02, -2.2909e-01,  4.6537e-02,
 6.5257e-02,  1.4804e-01, -6.2092e-02,
-1.5699e-02, -1.5303e-02,  1.6671e-01,
-6.1947e-03,  2.5749e-01,  1.5257e-01,
 3.2908e-02, -5.9907e-02,  1.1502e-01,
 7.5876e-02, -2.6699e-01, -1.5891e-02,
-8.0426e-02,  1.3406e-01, -1.9881e-02,
 3.5472e-02, -8.2140e-02,  1.6509e-02,
 8.3390e-03, -7.8291e-02, -2.0754e-01,
 3.4490e-02,  2.7913e-01,  5.9566e-02,
 2.5288e-02,  1.1725e-01, -1.0356e-01,
-5.0955e-02,  9.2093e-02, -5.8477e-02,
 4.4325e-02,  3.2973e-02, -1.9477e-01,
 3.9582e-02, -8.6877e-02, -1.1753e-01,
 3.0401e-02, -2.8757e-02, -2.5563e-02,
 5.0741e-02, -3.5056e-01, -2.5584e-01,
 9.1709e-02, -4.0932e-02,  2.3812e-01,
 5.0945e-02,  4.9246e-02,  1.2738e-01,
 5.1440e-03,  1.5703e-01,  5.5743e-02,
-3.9492e-02,  1.2114e-01,  2.0531e-02,
 8.0800e-02,  2.6680e-03, -1.6660e-02,
 1.0684e-01,  1.2308e-01,  1.7882e-02,
 1.8280e-02,  1.0972e-01, -5.2912e-03
}
,)"
R"(
{
-1.3812e-02, -4.6271e-02,  7.3790e-02,
-6.3801e-02, -3.6817e-01, -1.7880e-02,
 5.2986e-02,  1.8626e-01,  1.5645e-03,
 1.2367e-02, -6.2923e-02,  3.0844e-02,
 9.3623e-02,  1.9527e-01, -2.6366e-02,
-2.0837e-02, -3.4424e-02,  4.0256e-02,
 4.1482e-02,  6.1795e-02, -1.1293e-02,
-8.9944e-02, -1.3608e-01,  1.8067e-02,
 3.6974e-02,  5.2530e-03, -2.7474e-02,
 1.1872e-05,  1.9000e-05,  2.0729e-05,
 1.0139e-05,  1.6832e-05,  1.9392e-05,
 6.5445e-06,  1.0973e-05,  1.3521e-05,
-5.3340e-02,  1.3108e-03,  4.0436e-02,
 5.7068e-02, -2.7923e-02, -5.4781e-02,
-2.9293e-02,  2.7145e-02,  2.7340e-02,
 5.3520e-03,  1.8766e-02,  4.0297e-01,
 2.6473e-02, -3.4675e-02, -1.1783e-01,
-2.5038e-02, -1.7702e-02, -3.4908e-02,
 1.4847e-02,  2.3237e-01, -6.3687e-02,
-6.5672e-02, -2.1888e-01, -1.7233e-02,
 4.0608e-02, -6.9580e-02, -2.2200e-02,
 5.8163e-02,  1.3695e-01, -2.6257e-02,
-1.3328e-01, -3.5730e-01,  2.4507e-02,
-4.5611e-03,  2.0424e-01, -3.9821e-02,
 5.5300e-02, -1.6006e-01,  1.1717e-01,
-2.6107e-02, -8.6995e-02,  8.3720e-02,
 7.5494e-02,  3.2189e-01,  1.5527e-01,
-6.6869e-02,  1.4469e-01,  5.1805e-02,
 9.8760e-02, -1.6759e-01, -1.2350e-01,
 5.7005e-02,  8.4904e-02,  8.9713e-02,
-1.4263e-02,  2.8914e-02,  3.2239e-02,
-2.4871e-02,  5.6014e-02, -4.4469e-02,
 3.1209e-02,  1.3677e-02, -2.1052e-02,
-1.6548e-03, -1.8796e-03, -1.9883e-03,
-1.6186e-03, -1.8494e-03, -1.9670e-03,
-1.5841e-03, -1.8173e-03, -1.9345e-03,
 3.5726e-02,  1.8013e-01,  1.6913e-02,
-1.2168e-01, -6.3848e-02,  3.0555e-02,
 3.0269e-02, -1.0260e-01, -1.5259e-02,
-4.7375e-03,  5.5115e-02,  6.2642e-01,
 9.9776e-03, -2.1988e-01, -2.0984e-01,
 7.0470e-03,  6.3178e-02, -1.3607e-02,
 1.1918e-01, -2.4081e-01,  1.7889e-01,
-1.0514e-01,  2.9220e-01, -1.3263e-01,
 5.6091e-03, -4.1623e-02,  2.5589e-02,
-1.8496e-01,  2.7698e-02, -6.5768e-02,
 2.9677e-01,  4.4163e-02,  5.8530e-02,
-1.1010e-01, -7.6787e-02,  3.9844e-02,
 5.2113e-03, -1.8202e-02,  1.4129e-03,
-6.1402e-03, -2.7222e-01,  7.4690e-02,
 1.9131e-02,  2.2753e-01,  1.9587e-02,
-2.7391e-02,  6.7917e-03,  2.0496e-03,
 6.7333e-02,  7.8262e-02,  2.1110e-03,
-5.4519e-02,  3.0763e-02,  1.5628e-02,
 9.5055e-02,  3.8855e-02,  1.2446e-02,
-1.5152e-01,  7.8124e-02, -1.2616e-02,
 9.3100e-03, -1.6528e-02, -1.2873e-02,
-1.8377e-03, -1.9231e-03, -1.8930e-03,
-1.8058e-03, -1.8841e-03, -1.8678e-03,
-1.7387e-03, -1.7966e-03, -1.7781e-03,
-4.5122e-02,  1.7027e-03, -3.5534e-03,
 8.5222e-03,  1.0130e-01,  4.7893e-02,
 6.5574e-02,  7.2150e-03, -2.1820e-03,
-5.5105e-03, -1.8990e-01,  2.6527e-02,
 6.6140e-03,  2.1537e-01, -2.2183e-02,
-8.0628e-03,  6.8398e-03,  9.4474e-03,
 1.2239e-01, -1.3337e-01,  7.3391e-02,
-1.2205e-01,  1.3145e-01, -2.0063e-02,
 2.2168e-02,  3.6097e-03,  2.7146e-02,
 4.6717e-02,  2.1122e-02,  1.5491e-02,
-1.3077e-01,  1.1635e-01,  1.0849e-02,
 8.0113e-02, -8.4028e-02,  1.2863e-03,
-2.9796e-02, -8.4537e-02, -2.6766e-03,
-7.7771e-03, -2.4274e-03,  8.6274e-02,
-2.0354e-02,  4.1245e-02,  8.4227e-02,
 5.5894e-02,  1.0706e-01,  5.2965e-02,
-7.8731e-03,  5.5825e-01,  1.0373e-01,
-1.1975e-01, -2.0071e-02, -2.5286e-02,
-7.7477e-02,  5.3589e-02, -1.5710e-03,
-1.2753e-01,  2.5166e-01,  8.2205e-03,
-9.8349e-02, -4.9539e-02, -5.4941e-02,
-4.9916e-03, -4.9986e-03, -5.0660e-03,
-4.9770e-03, -4.9840e-03, -5.0543e-03,
-4.9997e-03, -5.0114e-03, -5.0809e-03,
 6.1819e-02,  1.5061e-01,  1.1984e-02,
 1.2905e-01,  2.5921e-01,  1.4768e-01,
 4.5548e-02,  1.4902e-01, -4.8961e-03,
-1.3605e-02,  8.2896e-02, -4.1931e-01,
-2.2657e-02,  2.4768e-01,  2.6528e-01,
-1.1566e-02, -8.7819e-03,  4.3618e-02,
-3.4332e-02, -1.8392e-01,  4.4471e-02,
-3.7073e-02, -5.4620e-02,  1.0899e-01,
 3.7891e-02,  9.9487e-02,  3.2383e-02,
-6.3628e-02, -5.0303e-03,  5.4617e-02,
-8.7802e-02,  2.1977e-01, -6.0249e-03,
 6.3554e-02, -5.4291e-02, -2.6709e-02,
-1.5505e-02, -6.7104e-02,  3.8607e-02,
-1.1427e-01, -3.2524e-01,  4.0077e-02,
-6.5144e-03,  1.2313e-01, -2.7924e-02,
 1.4265e-02, -3.8338e-02,  8.6780e-02,
 1.5341e-01,  1.2174e-01, -7.3160e-02,
 2.6326e-04,  7.3690e-02,  5.2187e-02,
-3.3114e-02, -3.6588e-02,  1.1635e-02,
-3.3521e-02,  1.0767e-01, -8.9125e-03,
-2.2431e-02, -4.5655e-03,  7.5531e-03,
 6.7227e-04,  7.2856e-04,  7.3907e-04,
 6.5335e-04,  7.0702e-04,  7.1233e-04,
 6.1540e-04,  6.7286e-04,  6.7797e-04,
-3.1496e-02,  6.0514e-02,  4.2013e-02,
-2.8617e-02,  1.4846e-02,  4.0016e-03,
 4.7006e-03, -4.0017e-02, -3.0411e-02,
-9.6037e-03,  8.8522e-02,  9.8616e-02,
 4.1297e-02, -3.2645e-01, -7.6144e-03,
-1.0711e-02,  3.9324e-02,  4.0144e-02,
 5.2899e-02, -7.8668e-02, -5.4798e-02,
-2.0428e-01,  5.7238e-02, -3.6937e-02,
-3.6103e-02, -8.2683e-02, -2.8101e-02,
 8.2479e-02,  5.7766e-02, -1.2019e-01,
-3.8373e-01,  6.8272e-02, -1.1758e-02,
 5.1129e-02, -2.7931e-01,  4.5608e-02,
-2.5151e-02, -5.0816e-02,  1.7231e-02,
-3.6376e-02,  1.5916e-01,  2.9192e-02,
-4.1947e-02,  5.3183e-02, -9.7289e-02,
 4.6138e-02,  7.0842e-02,  1.6673e-02,
-1.7243e-03,  2.7203e-01,  3.8262e-02,
-1.4000e-01, -7.3793e-02, -2.0050e-02,
-1.8750e-02, -8.5319e-02, -3.0858e-02,
-5.9981e-02,  1.2729e-01,  1.4094e-02,
-5.4088e-02, -2.3694e-02, -9.7485e-03,
-4.7840e-03, -4.8359e-03, -4.8727e-03,
-4.7882e-03, -4.8380e-03, -4.8755e-03,
-4.7859e-03, -4.8321e-03, -4.8633e-03,
 4.9511e-02,  1.0935e-01, -3.7430e-03,
 1.1834e-01,  7.7243e-02,  4.3074e-02,
 6.7446e-02,  2.9734e-02, -1.1276e-02,
-2.0080e-02,  1.3561e-01, -1.3455e-01,
-1.4505e-02,  2.2100e-01,  4.9635e-02,
-1.0040e-02,  3.4560e-02, -7.4607e-03,
-6.8873e-02, -5.6221e-02,  1.2255e-02,
-2.9198e-02,  7.1612e-02,  2.9402e-02,
 4.1036e-02,  4.6417e-02,  6.0284e-03,
-6.5261e-02,  2.1426e-03,  2.4192e-02,
-1.6073e-03, -6.2222e-03, -1.8295e-02,
 2.4952e-04, -2.0623e-02, -3.3064e-03,
 5.9188e-02, -4.8839e-02,  7.9840e-02,
-6.7952e-02, -4.7191e-01,  1.5117e-01,
 1.5668e-01,  2.4733e-01,  1.1354e-01,
 1.7742e-02, -4.4059e-02,  9.5374e-03,
 3.2049e-01, -1.3779e-01,  9.6608e-02,
 8.4580e-02,  1.4293e-01,  6.1574e-02,
 2.8777e-03,  7.8795e-02, -5.1902e-02,
 1.2212e-01,  1.0321e-01,  3.2360e-02,
-9.6617e-02,  7.8941e-03, -7.0876e-02,
 3.5869e-03,  3.5891e-03,  3.5923e-03,
 3.5746e-03,  3.5840e-03,  3.5967e-03,
 3.5785e-03,  3.5932e-03,  3.6080e-03,
 1.5454e-03,  3.0582e-03,  4.3737e-02,
-5.9833e-02, -1.1247e-01,  4.4380e-02,
-1.3206e-01,  8.2778e-03,  4.7963e-02,
-4.3720e-02, -7.5722e-03,  2.0510e-01,
 3.0133e-02, -4.0506e-01,  2.7867e-01,
 5.5586e-02,  2.8926e-02,  1.3360e-03,
 1.9490e-05,  3.3326e-01, -7.7241e-02,
-1.5648e-01,  1.5195e-01, -1.3995e-01,
 8.6519e-02,  1.0447e-01, -4.1413e-02,
-3.8667e-03,  1.6159e-01,  1.1627e-01,
-2.2646e-01, -3.4758e-02, -6.7956e-03,
-3.2689e-01,  1.9606e-01, -9.1523e-02,
 1.1238e-02,  1.5084e-03,  4.2113e-02,
-1.1154e-02, -3.6596e-01, -7.2252e-02,
 6.6621e-02,  1.0188e-01,  4.1032e-01,
 3.5892e-02, -4.8304e-02,  6.6142e-03,
 1.3374e-01,  2.2720e-01, -7.1224e-02,
 6.8952e-02,  2.0467e-01,  5.0251e-02,
-6.2016e-02,  2.2175e-01, -1.7764e-02,
 2.7542e-02,  1.4905e-01,  3.6637e-02,
-7.2231e-02,  5.0271e-03, -7.1823e-02,
 3.5760e-03,  3.5540e-03,  3.5692e-03,
 3.5664e-03,  3.5490e-03,  3.5689e-03,
 3.5671e-03,  3.5619e-03,  3.5864e-03,
 2.7470e-02, -3.9752e-02,  4.1063e-02,
-2.4985e-02, -1.7969e-01,  8.2186e-02,
-5.4251e-02, -5.9651e-03,  2.5079e-02,
-2.1197e-02,  2.5426e-02,  1.3585e-01,
-1.3460e-02, -1.1377e-01,  1.2278e-01,
 3.6533e-02,  1.2843e-02,  5.6219e-02,
 5.8141e-04,  2.8354e-01, -6.2016e-02,
-1.0289e-01,  1.8724e-01, -9.9475e-02,
 5.1193e-02,  7.5986e-02, -1.2951e-03,
-8.2587e-02,  1.8498e-01,  1.0891e-01,
 1.3538e-01, -4.7728e-01,  1.0868e-01,
-8.6415e-02, -1.7061e-01,  1.0457e-02
}
};
)"
R"(
__constant float biasL[8][8] = 
{
{
-0.1175, -0.0258, -0.0053, -0.0437, -0.0563, -0.1047, -0.3449,  0.0568
}
,
{
 0.0339, -0.1738,  0.0061,  0.1565, -0.0316, -0.0016, -0.0032, -0.0554
}
,
{
-0.0508, -0.0609,  0.0347, -0.0802, -0.0438,  0.2512, -0.0491, -0.0259
}
,
{
 0.0655,  0.0255,  0.0228, -0.0027, -0.0155, -0.0163, -0.0174, -0.1095
}
,
{
 4.9947e-03,  5.3372e-03, -4.5286e-09, -1.3756e-03,  3.8858e-03, -4.4197e-02,  3.3970e-02,  2.8411e-02
}
,
{
-0.0396,  0.0007,  0.1735,  0.0109,  0.1177,  0.0919,  0.0567, -0.0005
}
,
{
 0.0127, -0.0688,  0.1102, -0.0052,  0.1602, -0.0191, -0.0322,  0.0311
}
,
{
 0.0063, 0.0093, 0.0729, 0.3734, 0.0006, 0.1915, 0.3186, 0.2636
}
};

__constant float kernelsL10[4 * 8] = 
{
-0.0967, -0.3094,
 0.3537,  0.5705,
 0.2547,  0.3360,
-0.0718, -0.0700,
-0.3013, -0.1602,
 0.4520,  0.0495,
 0.1564,  0.3773,
-0.0216,  0.4367,
-0.4855, -0.1972,
-0.2026, -0.4390,
 0.3743, -0.1156,
 0.4408, -0.3123,
-0.3577,  0.0753,
-0.3396,  0.0336,
 0.1052, -0.4180,
 0.0799, -0.3587
};)") + kernelFunction
};
#endif // BUILT_IN_KERNEL
