#ifndef DS_ONNX_INFER_SPEAKEREMBED_H
#define DS_ONNX_INFER_SPEAKEREMBED_H

#include <array>
#include <string>
#include <unordered_map>

#include "TString.h"

namespace diffsinger {

    constexpr unsigned int SPK_EMBED_SIZE = 256;
    using SpeakerEmbedArray = std::array<float, SPK_EMBED_SIZE>;
    using SpeakerEmbedMap = std::unordered_map<std::string, SpeakerEmbedArray>;

    class SpeakerEmbed {
    private:
        SpeakerEmbedMap m_emb;
    public:
        SpeakerEmbed();
        SpeakerEmbed(const std::vector<std::string> &speakers, const TString &path);

        void loadSpeakers(const std::vector<std::string> &speakers, const TString &path);
        SpeakerEmbedArray getMixedEmb(const std::unordered_map<std::string, double> &mix) const;
        SpeakerEmbedArray getMixedEmb(const std::string &inputString) const;

        static std::unordered_map<std::string, double> parseMixString(const std::string &inputString);

        const SpeakerEmbedMap &getEmb();
    };
}

#endif //DS_ONNX_INFER_SPEAKEREMBED_H
