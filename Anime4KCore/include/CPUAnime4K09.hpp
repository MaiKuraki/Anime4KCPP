#pragma once

#include"FilterProcessor.hpp"

namespace Anime4KCPP
{
    namespace CPU
    {
        class DLL Anime4K09;
    }
}

class Anime4KCPP::CPU::Anime4K09 :public AC
{
public:
    Anime4K09(const Parameters& parameters = Parameters());
    virtual ~Anime4K09() = default;

    virtual std::string getInfo() override;
    virtual std::string getFiltersInfo() override;
private:
    virtual void processYUVImageB() override;
    virtual void processRGBImageB() override;
    virtual void processRGBVideoB() override;

    virtual void processYUVImageF() override;
    virtual void processRGBImageF() override;

    void getGrayB(cv::Mat& img);
    void pushColorB(cv::Mat& img);
    void getGradientB(cv::Mat& img);
    void pushGradientB(cv::Mat& img);
    void getGrayF(cv::Mat& img);
    void pushColorF(cv::Mat& img);
    void getGradientF(cv::Mat& img);
    void pushGradientF(cv::Mat& img);

    void changEachPixelBGRA(cv::Mat& src, const std::function<void(int, int, PixelB, LineB)>&& callBack);
    void changEachPixelBGRA(cv::Mat& src, const std::function<void(int, int, PixelF, LineF)>&& callBack);
    void getLightest(PixelB mc, const PixelB a, const PixelB b, const PixelB c) noexcept;
    void getLightest(PixelF mc, const PixelF a, const PixelF b, const PixelF c) noexcept;
    void getAverage(PixelB mc, const PixelB a, const PixelB b, const PixelB c) noexcept;
    void getAverage(PixelF mc, const PixelF a, const PixelF b, const PixelF c) noexcept;

    virtual Processor::Type getProcessorType() noexcept override;
};