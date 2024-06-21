package org.example.audioidentity.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
public class VoiceAuthService {

    @Value("${fastapi.url}")
    private String fastApiUrl;

    private final RestTemplate restTemplate;

    public VoiceAuthService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public Map<String, Object> authenticateVoice(byte[] voiceBytes) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(voiceBytes)
        { @Override public String getFilename()
        { return "Sample Audio File"; } });


        HttpEntity<MultiValueMap<String, Object>> requestEntity
                = new HttpEntity<>(body, headers);

        String serverUrl = fastApiUrl + "/predict/";
        RestTemplate restTemplate = new RestTemplate();
        try {
            return restTemplate.postForObject(serverUrl, requestEntity, Map.class);
        } catch (RestClientException e) {
            e.printStackTrace();
            throw new IllegalStateException();
        }


    }
}


