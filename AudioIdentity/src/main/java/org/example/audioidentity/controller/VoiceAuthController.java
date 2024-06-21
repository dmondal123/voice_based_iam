package org.example.audioidentity.controller;

import org.example.audioidentity.service.VoiceAuthService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

@RestController
public class VoiceAuthController {

    @Autowired
    private VoiceAuthService voiceAuthService;

    @PostMapping(value = "/predictAudio", produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> handleAuthenticationRequest(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return Collections.singletonMap("error", true);
        }

        try {
            Map<String, Object> returnMap = new HashMap<>();
            byte[] audioBytes = file.getBytes();
            returnMap.putAll(voiceAuthService.authenticateVoice(audioBytes));
            returnMap.put("error", false);
            return returnMap;
        } catch (Exception e) {
            e.printStackTrace();
            return Collections.singletonMap("error", true);
        }
    }
}
