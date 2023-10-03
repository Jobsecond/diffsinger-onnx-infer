#include <fstream>
#include <iostream>
#include <regex>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "ArrayUtil.hpp"
#include "DsProject.h"
#include "SpeakerEmbed.h"

namespace diffsinger {

    int pitchOffset(char pitch);

    std::vector<DsSegment> loadDsProject(const TString &dsFilePath, const std::string &spkMixStr) {
        std::ifstream dsFile(dsFilePath);

        if (!dsFile.is_open()) {
            std::cout << "Failed to open file!\n";
            return {};
        }

        rapidjson::IStreamWrapper streamWrapper(dsFile);
        rapidjson::Document data;
        data.ParseStream(streamWrapper);

        if (!data.IsArray()) {
            std::cout << "Invalid ds file format!\n";
            dsFile.close();
            return {};
        }

        std::vector<DsSegment> result;

        auto numSegments = data.Size();
        for (rapidjson::SizeType i = 0; i < numSegments; i++) {
            DsSegment dsSegment{};
            const auto &segment = data[i];
            if (!segment.IsObject()) {
                std::cout << "Segment at index " << i << " is not an object!\n";
                continue;
            }

            // TODO: ph_dur and f0 curve can be inferred using rhythmizers and autopitch models.
            //       In this case, these parameters can be omitted from .ds files, but note sequences
            //       must be supplied.
            if (!segment.HasMember("ph_seq")
                || !segment.HasMember("ph_dur")
                || !segment.HasMember("f0_seq")
                || !segment.HasMember("f0_timestep")) {
                std::cout << "Segment at index " << i
                          << " must contain required keys (ph_seq, ph_dur, f0_seq, f0_timestep)!\n";
                continue;
            }
            if (!segment["ph_seq"].IsString()
                || !segment["ph_dur"].IsString()
                || !segment["f0_seq"].IsString()
                || !(segment["f0_timestep"].IsNumber() || segment["f0_timestep"].IsString())) {
                std::cout << "Segment at index " << i
                          << " must contain valid keys (ph_seq, ph_dur, f0_seq, f0_timestep)!\n";
                continue;
            }

            dsSegment.ph_seq = splitString<std::string>(segment["ph_seq"].GetString());
            dsSegment.ph_dur = splitString<double>(segment["ph_dur"].GetString());

            // ph_num (word_div)
            if (segment.HasMember("ph_num") && segment["ph_num"].IsString()) {
                dsSegment.ph_num = splitString<int>(segment["ph_num"].GetString());
            }

            // note_seq
            if (segment.HasMember("note_seq") && segment["note_seq"].IsString()) {
                auto vs = splitString<std::string>(segment["note_seq"].GetString());
                dsSegment.note_seq.reserve(vs.size());
                for (const auto &s : vs) {
                    dsSegment.note_seq.emplace_back(noteNameToMidi(s));
                }
            }

            // note_dur
            if (segment.HasMember("note_dur") && segment["note_dur"].IsString()) {
                dsSegment.note_dur = splitString<double>(segment["note_dur"].GetString());
            }

            auto loadSampleCurve = [&segment](
                    const char *sampleKey, const char *timestepKey,
                    SampleCurve *sampleCurve) {
                if (!sampleCurve) {
                    return;
                }
                if (!segment.HasMember(sampleKey) || !segment.HasMember(timestepKey)) {
                    return;
                }
                if (!segment[sampleKey].IsString()) {
                    return;
                }
                bool timestepIsString = segment[timestepKey].IsString();
                bool timestepIsNumber = segment[timestepKey].IsNumber();
                if (!timestepIsString && !timestepIsNumber) {
                    return;
                }

                sampleCurve->samples = splitString<double>(segment[sampleKey].GetString());

                if (timestepIsString) {
                    sampleCurve->timestep = std::stod(segment[timestepKey].GetString());
                } else {
                    sampleCurve->timestep = segment[timestepKey].GetDouble();
                }
            };

            loadSampleCurve("f0_seq", "f0_timestep", &dsSegment.f0);
            loadSampleCurve("gender", "gender_timestep", &dsSegment.gender);
            loadSampleCurve("velocity", "velocity_timestep", &dsSegment.velocity);
            loadSampleCurve("energy", "energy_timestep", &dsSegment.energy);
            loadSampleCurve("breathiness", "breathiness_timestep", &dsSegment.breathiness);

            // TODO: Currently only supports static spk_mix.
            //       Haven't decided yet how to store curve spk_mix in ds file.
            if (!spkMixStr.empty()) {
                dsSegment.spk_mix = SpeakerMixCurve::fromStaticMix(SpeakerEmbed::parseMixString(spkMixStr));
            }

            dsSegment.offset = segment.HasMember("offset") ? segment["offset"].GetDouble() : 0.0;
            dsFile.close();

            result.push_back(dsSegment);
        }
        return result;
    }

    int pitchOffset(char pitch) {
        switch (pitch) {
            case 'C':
            case 'c':
                return 0;
            case 'D':
            case 'd':
                return 2;
            case 'E':
            case 'e':
                return 4;
            case 'F':
            case 'f':
                return 5;
            case 'G':
            case 'g':
                return 7;
            case 'A':
            case 'a':
                return 9;
            case 'B':
            case 'b':
                return 11;
            default:
                break;
        }
        return 0;
    }

    int noteNameToMidi(const std::string &note) {
        std::regex pattern(
                "\\s*"  // leading whitespaces (ignore)
                "([A-Ga-g])"  // note
                "([#b!]*)"  // accidental
                "([+-]?\\d+)?"  // octave
                "\\s*"  // trailing whitespaces (ignore)
        );
        std::smatch match;
        if (!std::regex_match(note, match, pattern)) {
            return 0;
        }
        std::string pitch = match[1].str();
        std::string accidental = match[2].str();
        std::string octave = (match.size() >= 4) ? match[3].str() : "0";

        if (pitch.empty()) {
            return 0;
        }

        int offset = 0;
        for (const auto &acc : accidental) {
            if (acc == '#') {
                ++offset;
            }
            else if ((acc == 'b') || (acc == '!')) {
                --offset;
            }
        }

        int octaveVal = octave.empty() ? 0 : std::stoi(octave);

        int midi = 12 * (octaveVal + 1) + pitchOffset(pitch[0]) + offset;
        return midi;
    }
}
