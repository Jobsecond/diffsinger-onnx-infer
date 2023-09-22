#include <fstream>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "ArrayUtil.hpp"
#include "DsProject.h"

namespace diffsinger {
    std::vector<DsSegment> loadDsProject(const TString &dsFilePath) {
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

            // TODO: spk_mix

            dsSegment.offset = segment.HasMember("offset") ? segment["offset"].GetDouble() : 0.0;
            dsFile.close();

            result.push_back(dsSegment);
        }
        return result;
    }
}
